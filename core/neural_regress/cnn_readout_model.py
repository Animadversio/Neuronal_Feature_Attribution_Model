
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import DataLoader
from collections import OrderedDict


class MultilayerCnn(nn.Module):
    """ CNN translated from 'Gaudy Image' paper
    https://github.com/pillowlab/gaudy-images/blob/a2948911f6b435096b819dad042b54d5170a8c8b/fig2/3layerconvnet_relu/class_model.py#L81
    """
    def __init__(self, in_channels, map_size=(7, 7), nonlinearity="relu"):
        super(MultilayerCnn, self).__init__()
        if nonlinearity == "relu":
            self.activation_func = nn.ReLU
        elif nonlinearity == "sigmoid":
            self.activation_func = nn.Sigmoid
        elif nonlinearity == "linear":
            self.activation_func = nn.Identity
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        self.in_channels = in_channels
        self.map_size = map_size
        # self.num_output_vars = 1
        self.model = nn.Sequential(OrderedDict([
            ('layer0_bn', nn.BatchNorm2d(num_features=in_channels)),
            ('initial_conv_layer', SeparableConv2D(in_channels=in_channels, out_channels=512, kernel_size=(1, 1), stride=1, padding='same', )),
            ('layer1_conv', SeparableConv2D(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding='same', )),
            ('layer1_bn', nn.BatchNorm2d(num_features=512)),
            ('layer1_act', self.activation_func()),
            ('layer2_conv', SeparableConv2D(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding='same', )),
            ('layer2_bn', nn.BatchNorm2d(num_features=512)),
            ('layer2_act', self.activation_func()),
            ('layer3_conv', SeparableConv2D(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding='same', )),
            ('layer3_bn', nn.BatchNorm2d(num_features=512)),
            ('layer3_act', self.activation_func()),
            # reshift matrices to take weighted average of spatial maps
            # ('final_spatial_pool', DepthwiseConv2D(kernel_size=(7, 7), strides=1, padding='valid', )),
            ('final_spatial_pool', nn.Conv2d(in_channels=512, out_channels=512, groups=512, kernel_size=map_size,
                                             stride=1, padding='valid', )),
            ('embeddings', nn.Flatten(start_dim=1)),
            ('Linear', nn.Linear(512, 1))]))

    def forward(self, x):
        return self.model(x)


class SeparableConv2D(nn.Module):
    """ Separable Convolutional Layer from
    https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=False):
        super(SeparableConv2D, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,
                                    padding=padding, stride=stride, bias=bias)
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FactorizedConv2D(nn.Module):
    """ Factorized Convolutional Layer from Klimt et al. (2018) """

    def __init__(self, in_channels, out_channels, kernel_size, factors, bias=True, bn=True): # stride=1, padding='same',
        super(FactorizedConv2D, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels) if bn else nn.Identity()
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=factors, kernel_size=(1, 1), bias=False)
        self.depth_conv = nn.Conv2d(in_channels=factors, out_channels=factors, kernel_size=kernel_size, groups=factors, bias=False)
        self.linear = nn.Linear(in_features=factors, out_features=out_channels, bias=bias)
        nn.init.xavier_uniform_(self.point_conv.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.depth_conv.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        x = self.bn(x)
        x = self.point_conv(x)
        x = self.depth_conv(x)
        x = x.flatten(start_dim=1)
        # x = nn.ReLU()(x)
        x = self.linear(x)
        return x


#%%
if __name__ is "__main__":
    from torchsummary import summary
    model = MultilayerCnn(1024)
    summary(model, input_size=(1024, 7, 7), device="cpu")
    #%%
    model = MultilayerCnn(1024, map_size=(14, 14))
    summary(model, input_size=(1024, 14, 14), device="cpu")
    #%%
    summary(FactorizedConv2D(1024, 1, (14, 14), 3),
            input_size=(1024, 14, 14), device="cpu")
    #%%
    summary(FactorizedConv2D(2048, 1, (7, 7), 3),
            input_size=(2048, 7, 7), device="cpu")

    # nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))