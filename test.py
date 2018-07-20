from __future__ import division, print_function
import torch
from torch.autograd import Variable
from coord_conv import AddCoordinates, CoordConv, CoordConvTranspose, \
    CoordConvNet
import pdb


def generate_input(batch_size, image_height, image_width, usegpu=False):

    if usegpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    input_image = torch.rand(batch_size, 3, image_height, image_width)
    input_image.to(device)

    return input_image


def test_addCoordinates(input_image):

    print('- AddCoordinates')

    coord_adder = AddCoordinates(with_r=True)

    output = coord_adder(input_image)

    print('* Y :\n', output[0, 0])
    print('* X :\n', output[0, 1])
    print('* R :\n', output[0, 2])

    print('- AddCoordinates: OK!')


def test_coordConv(input_image):

    print('- CoordConv')

    coord_conv = CoordConv(3, 64, 3, with_r=True)
    output = coord_conv(input_image)

    print('Input Size  : ', input_image.size())
    print('Output Size : ', output.size())

    print('- CoordConv: OK!')


def test_coordConvTranspose(input_image):

    print('- CoordConvTranspose')

    coord_conv_transpose = CoordConvTranspose(3, 64, 3, with_r=True)
    output = coord_conv_transpose(input_image)

    print('Input Size  : ', input_image.size())
    print('Output Size : ', output.size())

    print('- CoordConvTranspose: OK!')


def test_coordConvNet():

    print('- CoordConvNet')

    import torchvision.models as models

    vgg16 = models.__dict__['vgg16'](pretrained=False)

    print('VGG16 :\n', vgg16)

    vgg16 = CoordConvNet(vgg16, with_r=True)

    print('CoordVGG16 :\n', vgg16)

    output = vgg16(input_image)

    print('Input Size  : ', input_image.size())
    print('Output Size : ', [i.size() for i in output])

    print('- CoordConvNet: OK!')

if __name__ == '__main__':

    usegpu = True
    input_image = generate_input(2, 64, 64, usegpu=usegpu)

    test_addCoordinates(input_image)
    test_coordConv(input_image)
    test_coordConvTranspose(input_image)
    test_coordConvNet()
