import torch
from torch.autograd import Variable
from coord_conv import AddCoordinates, CoordConv, CoordConvTranspose, \
    CoordConvNet


def generate_input(batch_size, image_height, image_width, usegpu=False):

    input_image = torch.rand(batch_size, 3, image_height, image_width)
    input_image = Variable(input_image)
    if usegpu:
        input_image = input_image.cuda()

    return input_image


def test_addCoordinates(input_image, usegpu=False):

    print '- AddCoordinates'

    coord_adder = AddCoordinates(with_r=True, usegpu=usegpu)

    output = coord_adder(input_image)

    print '* Y :\n', output[0, 0]
    print '* X :\n', output[0, 1]
    print '* R :\n', output[0, 2]

    print '- AddCoordinates: OK!'


def test_coordConv(input_image, usegpu=False):

    print '- CoordConv'

    coord_conv = CoordConv(3, 64, 3, with_r=True, usegpu=usegpu)
    if usegpu:
        coord_conv = coord_conv.cuda()
    output = coord_conv(input_image)

    print 'Input Size  : ', input_image.size()
    print 'Output Size : ', output.size()

    print '- CoordConv: OK!'


def test_coordConvTranspose(input_image, usegpu=False):

    print '- CoordConvTranspose'

    coord_conv_transpose = CoordConvTranspose(3, 64, 3, with_r=True,
                                              usegpu=usegpu)
    if usegpu:
        coord_conv_transpose = coord_conv_transpose.cuda()
    output = coord_conv_transpose(input_image)

    print 'Input Size  : ', input_image.size()
    print 'Output Size : ', output.size()

    print '- CoordConvTranspose: OK!'


def test_coordConvNet(usegpu=False):

    print '- CoordConvNet'

    import torchvision.models as models

    vgg16 = models.__dict__['vgg16'](pretrained=False)

    print 'VGG16 :\n', vgg16

    vgg16 = CoordConvNet(vgg16, with_r=True, usegpu=usegpu)

    print 'CoordVGG16 :\n', vgg16

    if usegpu:
        vgg16 = vgg16.cuda()

    output = vgg16(input_image)

    print 'Input Size  : ', input_image.size()
    print 'Output Size : ', output.size()

    print '- CoordConvNet: OK!'

if __name__ == '__main__':
    usegpu = False

    input_image = generate_input(2, 64, 64, usegpu=usegpu)

    test_addCoordinates(input_image, usegpu)
    test_coordConv(input_image, usegpu)
    test_coordConvTranspose(input_image, usegpu)
    test_coordConvNet(usegpu)
