# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset


class MRIdataset(Dataset):
    def __init__(self, csv_file, descriptor, root_dir, image_size):
        super(MRIdataset, self).__init__()
        self.root_dir = root_dir + '/Processed'
        self.image_size = np.asarray(image_size)

        self.task = -2  # -2=cancer; -1=nodule/mass

        labels = pd.read_csv(csv_file)
        self.labels = labels.set_index('patient_id').T.to_dict('list')
        descriptor = pd.read_csv(descriptor)
        self.descriptor = descriptor.set_index('patient_id').T.to_dict('list')
        self.idx = list(self.labels.keys())
        sizes = pd.read_csv('../Data/sizes.csv')
        sizes = sizes.set_index('access').T.to_dict('list')
        for i in sizes:
            if int(sizes[i][0]) < 250 and i in self.idx:
                self.idx.remove(i)
        self.weights = self.weights_balanced()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        patient = self.idx[idx]
        label = self.labels[patient][self.task]
        descriptor = self.descriptor[patient]
        descriptor = descriptor[1:12] + descriptor[13:-1]
        descriptor = np.array(descriptor, dtype=np.float32)
        image = load_image(str(patient) + '.nii.gz', self.root_dir)

        if any(np.asarray(image.shape) <= self.image_size):
            dif = self.image_size - image.shape
            mod = dif % 2
            dif = dif // 2
            pad = np.maximum(dif, [0, 0, 0])
            pad = tuple(zip(pad, pad + mod))
            image = np.pad(image, pad, 'reflect')

        sz = self.image_size[0]
        if any(np.asarray(image.shape) >= self.image_size):
            x, y, z = image.shape
            x = x // 2 - (sz // 2)
            y = y // 2 - (sz // 2)
            z = z // 2 - (sz // 2)
            image = image[x:x + sz, y:y + sz, z:z + sz]
        # Stats obtained from the MSD dataset
        image = np.clip(image, a_min=-1024, a_max=326)
        image = (image - 159.14433291523548) / 323.0573880113456

        return {'data': np.expand_dims(image, 0),
                'descriptor': descriptor, 'target': label, 'id':patient}

    def weights_balanced(self):
        count = [0] * 2
        for item in self.idx:
            count[self.labels[item][self.task]] += 1
        weight_per_class = [0.] * 2
        N = float(sum(count))
        for i in range(2):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(self.idx)
        for idx, val in enumerate(self.idx):
            weight[idx] = weight_per_class[self.labels[val][self.task]]
        return weight


class collate(object):
    def __init__(self, size):
        self.transforms = DA.Compose([
            DA.NumpyToTensor()])

    def __call__(self, batch):
        elem = batch[0]
        batch = {key: np.stack([d[key] for d in batch]) for key in elem}
        return self.transforms(**batch)


def load_image(patient, root_dir):
    im = nib.load(os.path.join(root_dir, patient))
    image = im.get_fdata()
    return image
