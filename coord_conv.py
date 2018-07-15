import torch
from torch import nn
from torch.autograd import Variable

class AddCoordinates(object):
    r"""Coordinate Adder Module.

    This module concatenates coordinate information (x, y, and r) with given
    input tensor.

    x and y coordinates are scaled to [-1, 1] range where origin is the center.
    r is the Euclidean distance from the center and is scaled to [0, 1].

    Input tensor shape : [N, C, H, W]
    """

    def __init__(self, with_r=False, usegpu=True):
        
        self.with_r = with_r
        self.usegpu =  usegpu

    def __call__(self, image):

        batch_size, _, image_height, image_width = image.size()

        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)

        coords = Variable(coords)

        if self.usegpu:
            coords = coords.cuda()

        image = torch.cat((coords, image), dim=1)

        return image


class CoordConv(nn.Module):
    r"""2D Convolution Module Using Extra Coordinate Information."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r=False, usegpu=True):
        super(CoordConv, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=1, padding=0,
                                    dilation=1, groups=1, bias=True)

        self.coord_adder = AddCoordinates(with_r, usegpu)

    def forward(self, x):

        x = self.coord_adder(x)
        x = self.conv_layer(x)
 
        return x


class CoordConvNet(nn.Module):
    r"""Improves 2D Convolutions inside a ConvNet by processing extra
    coordinate information.

    This module adds coordinate information to inputs of each 2D convolution
    module (torch.nn.Conv2d).

    Assumption: ConvNet Model must contain single Sequential container
    (torch.nn.modules.container.Sequential)."""

    def __init__(self, cnn_model, with_r=False, usegpu=True):
        super(CoordConvNet, self).__init__()

        self.with_r = with_r

        self.cnn_model = cnn_model
        self.__get_model()
        self.__update_weights()

        self.coord_adder = AddCoordinates(self.with_r, usegpu)

    def __get_model(self):

        for module in list(self.cnn_model.modules()):
            if module.__class__ == torch.nn.modules.container.Sequential:
                self.cnn_model = module
                break

    def __update_weights(self):

        coord_channels = 2
        if self.with_r:
            coord_channels += 1

        for l in list(self.cnn_model.modules()):
            if l.__str__().startswith('Conv2d'):
                weights = l.weight.data

                out_channels, in_channels, k_height, k_width = weights.size()
        
                coord_weights = torch.zeros(out_channels, coord_channels,
                                            k_height, k_width)

                weights = torch.cat((coord_weights, weights), dim=1)
                weights = nn.Parameter(weights)

                l.weight = weights
                l.in_channels += coord_channels

    def forward(self, x):

        for layer_name, layer in self.cnn_model._modules.items():
            if layer.__str__().startswith('Conv2d'):
                x = self.coord_adder(x)
            x = layer(x)

        return x
