import torch
import torch.nn as nn
from .backbone import AlignedXception


class DeepLab(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLab, self).__init__()
        # Images
        BatchNorm = nn.InstanceNorm3d
        filters = [32, 64, 128, 256, 256, 512]
        self.backbone = AlignedXception(BatchNorm, filters)

        # Descriptor
        self.fc_d = nn.Linear(76, 512)

        # Combination
        self._fc0 = nn.Linear(filters[-1] * 4 * 4 * 4 + 512, filters[-1])
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(filters[-1], num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # Images
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)

        # Descriptor
        y = self.relu(self.fc_d(y))

        # Combination
        x = self.relu(self._fc0(torch.cat([x, y], dim=1)))
        x = self._dropout(x)
        x = self._fc(x)
        return x


if __name__ == "__main__":
    model = DeepLab()
    model.eval()
    input = torch.rand(1, 1, 256, 256, 256)
    output = model(input)
    print(output.size())
