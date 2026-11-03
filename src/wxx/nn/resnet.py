"""ResNet on ``wxx`` (ResNet-18 style)."""

from .module import (
    AdaptiveAvgPool2d,
    BatchNorm,
    Conv2D,
    Linear,
    MaxPool2d,
    Module,
    Sequential,
)


class ResBlock(Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = Conv2D(in_channel, out_channel, kernel=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm(out_channel)
        self.conv2 = Conv2D(out_channel, out_channel, kernel=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(out_channel)

        self.shortcut = Sequential()
        self.sh = False
        if stride != 1 or in_channel != out_channel:
            self.shortcut = Sequential(
                Conv2D(in_channel, out_channel, kernel=1, stride=stride, bias=False),
                BatchNorm(out_channel),
            )
            self.sh = True

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out).relu() if hasattr(self.bn1(out), "relu") else self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        return out.relu()


class ResNet18(Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.in_channels = 64
        self.conv1 = Conv2D(3, 64, kernel=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.maxpool = MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, out_channels, stride=1))
        return Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = out.relu()
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.reshape((out.shape[0], out.shape[1]))
        out = self.fc(out)
        return out


__all__ = ["ResNet18", "ResBlock"]
