import torch.nn as nn
from efficientnet_pytorch_3d import EfficientNet3D as EfficientNet

'''
efficientnet_pytorch_3d: https://github.com/shijianjian/EfficientNet-PyTorch-3D
install efficientnet_pytorch_3d by pip using:
pip install git+https://github.com/shijianjian/EfficientNet-PyTorch-3D
'''

class EfficientNet3D(nn.Module):
    def __init__(self, version='efficientnet-b0', drop_rate=0.5, num_classes=1):
        super(EfficientNet3D, self).__init__()
        self.model = EfficientNet.from_name("efficientnet-b0",
                                            override_params={'num_classes': 1},
                                            in_channels=1)
        self.model._fc = nn.Sequential(
            nn.Linear(self.model._fc.in_features, 1024),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x
