import os
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def save_graph(folder):
    info = reader(os.path.join(folder, 'progress.csv'))

    name = folder + '/Progress.png'
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(info[0], info[1], label='Train')
    ax1.plot(info[0], info[2], label='Val')
    ax1.set_ylim(top=3.5, bottom=0)
    ax1.legend()
    ax1.grid()
    ax1.set(title='Prueba ' + folder)

    ax2 = ax1.twinx()

    ax2.set_ylabel('Accuracy')
    ax2.plot(info[0], info[3], 'g', label='Nodule')
    # ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(top=1, bottom=0)
    ax2.yaxis.grid(linestyle=(0, (1, 10)), linewidth=0.5)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(name, dpi=300)

    plt.close('all')


def reader(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        epoch = []
        lossT = []
        lossV = []
        accM = []
        for row in csv_reader:
            epoch.append(int(row[0]))
            lossT.append(float(row[1]))
            lossV.append(float(row[2]))
            accM.append(float(row[3]))
    return epoch, lossT, lossV, accM
