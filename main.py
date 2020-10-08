# -*- coding: utf-8 -*-
import os
import csv
import time
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import util.utils as utils
import util.Read_data as Read_data
from util.save_graphs import save_graph
from modeling.model import DeepLab

torch.autograd.set_detect_anomaly(True)


def main():
    # SET THE PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10,
                        help='lr scheduler patience (default: 10)')
    parser.add_argument('--batch', type=int, default=13,
                        help='Batch size (default: 13)')
    parser.add_argument('--name', type=str, default='Prueba',
                        help='Name of the current test (default: Prueba)')

    parser.add_argument('--load_model', type=str, default='best_f1',
                        help='Weights to load (default: best_f1)')
    parser.add_argument('--test', action='store_false', default=True,
                        help='Only test the model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Continue training a model')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Name of the folder with the pretrained model')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='Fine-tune a model')
    parser.add_argument('--freeze', action='store_false', default=True,
                        help='Freeze weights of the model')

    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU(s) to use (default: 0)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Train with automatic mixed precision')
    args = parser.parse_args()

    training = args.test
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.ft:
        args.resume = True

    args.image_size = [256, 256, 256]
    args.num_classes = 2

    # PATHS AND DIRS
    save_path = os.path.join('TRAIN', args.name)
    load_path = save_path
    if args.load_path is not None:
        load_path = os.path.join('TRAIN/', args.load_path)

    # DATA
    root = '../Data'  # Root directory to the data
    train_file = os.path.join(root, 'train_patients.csv')
    test_file = os.path.join(root, 'test_patients.csv')
    train_des = os.path.join(root, 'train_descriptor.csv')
    test_des = os.path.join(root, 'test_descriptor.csv')

    os.makedirs(save_path, exist_ok=True)

    # SEEDS
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # CREATE THE NETWORK ARCHITECTURE
    model = DeepLab(num_classes=args.num_classes)
    print('---> Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=1e-5, amsgrad=True)

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    annealing = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience, verbose=True)
    bce = nn.BCEWithLogitsLoss()

    # LOAD A MODEL IF NEEDED (TESTING OR CONTINUE TRAINING)
    args.epoch = 0
    best_f1 = 0
    if args.resume or not training:
        name = 'epoch_' + args.load_model + '.pth.tar'
        checkpoint = torch.load(
            os.path.join(load_path, name),
            map_location=lambda storage, loc: storage)
        args.lr = checkpoint['lr']

        print('Loading model and optimizer {}.'.format(checkpoint['epoch']))

        if args.amp:
            amp.load_state_dict(checkpoint['amp'])
        model.load_state_dict(checkpoint['state_dict'], strict=(not args.ft))
        if not args.ft:
            best_f1 = checkpoint['best_f1']
            args.epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

        if args.freeze:
            print('- Frozen Backbone -')
            for param in model.backbone.parameters():
                param.requires_grad = False

    # DATALOADERS
    train_data = Read_data.MRIdataset(train_file, train_des, root,
                                      args.image_size)
    test_data = Read_data.MRIdataset(test_file, test_des, root,
                                     args.image_size)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        train_data.weights, len(train_data.weights))
    train_loader = DataLoader(train_data, sampler=sampler, shuffle=True,
                              batch_size=args.batch, num_workers=20)
    test_loader = DataLoader(test_data, shuffle=False, sampler=None,
                             batch_size=args.batch, num_workers=20)

    # TRAIN THE MODEL
    is_best = True
    if training:
        torch.cuda.empty_cache()
        out_file = open(os.path.join(save_path, 'progress.csv'), 'a+')

        for epoch in range(args.epoch + 1, args.epochs + 1):
            args.epoch = epoch
            lr = utils.get_lr(optimizer)
            print('--------- Starting Epoch {} --> {} ---------'.format(
                epoch, time.strftime("%H:%M:%S")))
            print('Learning rate:', lr)

            train_loss = train(args, model, train_loader, optimizer, bce)
            test_loss, f1, flag = test(args, model, test_loader, bce)

            out_file.write('{},{},{},{},{}\n'.format(
                args.epoch, train_loss, test_loss, f1, lr))
            out_file.flush()

            annealing.step(test_loss)
            save_graph(save_path)

            # To avoid saving as "the best model" one that always predict the
            # same category
            is_best = False
            if flag:
                is_best = best_f1 < f1
                best_f1 = max(best_f1, f1)

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': [train_loss, test_loss],
                'lr': lr,
                'f1': f1,
                'best_f1': best_f1}
            if args.amp:
                state['amp'] = amp.state_dict()

            checkpoint = epoch % 50 == 0
            utils.save_epoch(state, save_path, epoch,
                             checkpoint=checkpoint, is_best=is_best)

            if lr <= (args.lr / (10 ** 4)):
                print('Stopping training: learning rate is too small')
                break
        out_file.close()

    # TEST THE MODEL
    if not is_best:
        checkpoint = torch.load(
            os.path.join(save_path, 'epoch_best_f1.pth.tar'),
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if args.amp:
            model = amp.initialize(model, opt_level="O1")
        print('Testing epoch with best f1 ({}: f1 {})'.format(
            checkpoint['epoch'], checkpoint['f1']))

    val_loss, flag = test(args, model, test_loader, save_path, bce, False)
    save_graph(save_path)


def train(args, model, loader, optimizer, bce):
    model.train()
    epoch_loss = utils.AverageMeter()
    batch_loss = utils.AverageMeter()

    print_stats = len(loader) // 5
    for batch_idx, sample in enumerate(loader):
        data = sample['data'].float().cuda()
        descriptor = sample['descriptor'].float().cuda()
        target = sample['target'].float().cuda()
        target = torch.stack([1 - target, target], dim=1)

        optimizer.zero_grad()
        out = model(data, descriptor)
        loss = bce(out, target)
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        batch_loss.update(loss.item())
        epoch_loss.update(loss.item())

        if batch_loss.count % print_stats == 0:
            text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(text.format(
                time.strftime("%H:%M:%S"), (batch_idx + 1),
                (len(loader)), 100. * (batch_idx + 1) / (len(loader)),
                batch_loss.avg))
            batch_loss.reset()
    print('--- Train: \tLoss: {:.6f} ---'.format(epoch_loss.avg))
    return epoch_loss.avg


def test(args, model, loader, save_path, bce, training=True):
    model.eval()
    epoch_loss = utils.AverageMeter()
    count, correct = 0, 0
    labels, patients, scores, predictions = [], [], [], []

    for batch_idx, sample in enumerate(loader):
        data = sample['data'].float().cuda()
        descriptor = sample['descriptor'].float().cuda()
        target = sample['target'].float().cuda()
        patients.extend(sample['id'].tolist())
        labels.extend(sample['target'].tolist())

        with torch.no_grad():
            out = model(data, descriptor)
        loss = bce(out, torch.stack([1 - target, target], dim=1))
        epoch_loss.update(loss.item())

        confidence = F.softmax(out, dim=1)
        scores.extend(confidence[:, 1].tolist())

        pred = torch.argmax(confidence, dim=1)
        predictions.extend(pred.tolist())
        count += pred.sum()
        correct += (pred * target).sum()

    print('--- Val: \tLoss: {:.6f} ---'.format(epoch_loss.avg))

    # Metrics
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    f1 = f1_score(labels, scores)

    if not training:
        print('ROC', roc, 'AP', ap, 'F1', f1)
        rows = zip(patients, scores)
        with open(os.path.join(save_path, 'confidence.csv'), "w") as f:
            writer = csv.writer(f)
            writer.writerow(['ROC:', roc])
            writer.writerow(['AP:', ap])
            writer.writerow(['F1:', f1])
            for row in rows:
                writer.writerow(row)

    count = count.sum()
    flag = True
    if count == 0 or count == len(loader.dataset):
        flag = False
    return epoch_loss.avg, f1, flag


if __name__ == '__main__':
    main()
