from __future__ import print_function

import math
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        self.Conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(out_planes)
        self.Conv2 = nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(self.expansion * out_planes)

        if stride != 1 or in_planes != self.expansion * out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes),
            )

    def forward(self, x):

        residual = self.downsample(x) if hasattr(self, 'downsample') else x

        output = self.Conv1(x)
        output = F.relu(self.BN1(output))

        output = self.Conv2(output)
        output = self.BN2(output)

        output += residual
        output = F.relu(output)

        return output


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(Bottleneck, self).__init__()

        self.Conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.BN1 = nn.BatchNorm2d(out_planes)
        self.Conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(out_planes)
        self.Conv3 = nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=1, stride=1, bias=False)
        self.BN3 = nn.BatchNorm2d(self.expansion * out_planes)

        if stride != 1 or in_planes != self.expansion * out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes),
            )

    def forward(self, x):

        residual = self.downsample(x) if hasattr(self, 'downsample') else x

        output = self.Conv1(x)
        output = F.relu(self.BN1(output))

        output = self.Conv2(output)
        output = F.relu(self.BN2(output))

        output = self.Conv3(output)
        output = self.BN3(output)

        output += residual
        output = F.relu(output)

        return output


class ResNet(nn.Module):
    """
    pre activated version of resnet
    """

    def __init__(self, block, layers, num_classes=25):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.Conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.BN1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_planes, blocks, stride=1):
        layers = []
        layers.append(block(self.in_planes, out_planes, stride))
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, out_planes))
            self.in_planes = out_planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.Conv1(x)
        output = F.relu(self.BN1(output))

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = output.view(x.size(0), -1)
        output = self.fc(output)

        return output

    def get_feature(self, x):
        output = self.Conv1(x)
        output = F.relu(self.BN1(output))

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = output.view(x.size(0), -1)

        return output

class ResNetWithDropout(ResNet):
    def __init__(self, block, layers, p=0.5, num_classes=25, featureModel=False):
        super(ResNetWithDropout, self).__init__(block, layers, num_classes=num_classes)

        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.dropout3 = nn.Dropout(p)
        self.dropout4 = nn.Dropout(p)
        self.featureModel = featureModel

    def forward(self, x):
        output = self.Conv1(x)
        output = F.relu(self.BN1(output))

        output = self.layer1(output)
        output = self.dropout1(output)
        output = self.layer2(output)
        output = self.dropout2(output)
        output = self.layer3(output)
        output = self.dropout3(output)
        output = self.layer4(output)
        output = self.dropout4(output)

        output = self.avgpool(output)
        output = output.view(x.size(0), -1)
        if not self.featureModel:
            output = self.fc(output)

        return output

def resnet_18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model
 vc

def resnet_34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


def resnet_50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def resnet_101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


def resnet_152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model

def resnet_dropout_18(num_classes=25, featureModel=False, p=0.5):
    model = ResNetWithDropout(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, featureModel=featureModel, p=p)
    return model