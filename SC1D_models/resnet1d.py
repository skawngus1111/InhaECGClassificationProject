"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BasicBlock(nn.Module) :
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module) :
    def __init__(self, n_lead, block, num_block, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv1d(n_lead, self.in_channels, kernel_size=13, padding=2, stride=2, bias=False)

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        # self.output_layer = nn.Linear(2048 * 128, num_classes)
        self.output_layer = nn.Linear(512 * 256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides :
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = output.view(output.size(0), -1)
        output = self.output_layer(output)

        return output

def resnet1d_18(input_dim, num_classes) :
    """ return a ResNet 34 object
    """
    return ResNet(input_dim, BasicBlock, [2, 2, 2, 2], num_classes)

def resnet1d_34(input_dim, num_classes) :
    """ return a ResNet 34 object
    """
    return ResNet(input_dim, BasicBlock, [2, 4, 6, 3], num_classes)

def resnet1d_50(input_dim, num_classes) :
    """ return a ResNet 34 object
    """
    return ResNet(input_dim, BottleNeck, [2, 4, 6, 3], num_classes)

def resnet1d_101(input_dim, num_classes) :
    """ return a ResNet 34 object
    """
    return ResNet(input_dim, BottleNeck, [3, 4, 23, 3], num_classes)

def resnet1d_152(input_dim, num_classes) :
    """ return a ResNet 34 object
    """
    return ResNet(input_dim, BottleNeck, [3, 8, 36, 3], num_classes)