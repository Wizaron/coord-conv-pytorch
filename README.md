# An intriguing failing of convolutional neural networks and the CoordConv solution

This repository implements CoordConv Module in [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247).

Blog post can be found [here](https://eng.uber.com/coordconv/).

`coord_conv.py` contains the modules and `test.py` includes methods to show usage of the modules.

## AddCoordinates

```
    Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).

    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.

    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.

    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`

    Examples:
        >>> coord_adder = AddCoordinates(True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)

        >>> coord_adder = AddCoordinates(True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_adder(input)

```

## CoordConv

```
    2D Convolution Module Using Extra Coordinate Information as defined
    in 'An Intriguing Failing of Convolutional Neural Networks and the
    CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).

    Args:
        Same as `torch.nn.Conv2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True, usegpu=False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv(input)

        >>> coord_conv = CoordConv(3, 16, 3, with_r=True, usegpu=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv(input)
```

## CoordConvTranspose

```
    2D Transposed Convolution Module Using Extra Coordinate Information
    as defined in 'An Intriguing Failing of Convolutional Neural Networks and
    the CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).

    Args:
        Same as `torch.nn.ConvTranspose2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True,
        >>>                                    usegpu=False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv_tr(input)

        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True,
        >>>                                    usegpu=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv_tr(input)
```

## CoodConvNet

```
    Improves 2D Convolutions inside a ConvNet by processing extra
    coordinate information as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).

    This module adds coordinate information to inputs of each 2D convolution
    module (`torch.nn.Conv2d`).

    Assumption: ConvNet Model must contain single `Sequential` container
    (`torch.nn.modules.container.Sequential`).

    Args:
        cnn_model: A ConvNet model that must contain single `Sequential`
            container (`torch.nn.modules.container.Sequential`).
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: Same as the input of the model.
        - Output: A list that contains all outputs (including
            intermediate outputs) of the model.

    Examples:
        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet(cnn_model, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = cnn_model(input)

        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet(cnn_model, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = cnn_model(input)
```

## Environment

* Python version : 2.7
* PyTorch version : 0.4.0
* Torchvision version : 0.2.1
