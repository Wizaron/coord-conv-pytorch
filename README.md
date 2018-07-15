# An intriguing failing of convolutional neural networks and the CoordConv solution

This repository implements CoordConv Module in [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247).

`coord_conv.py` contains the modules and `test.py` includes methods to show usage of the modules.

* `AddCoordinates` : This module concatenates coordinate information (`x`, `y`, and `r`) with given input tensor. `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the **center**. `r` is the Euclidean distance from the **center** and is scaled to `[0, 1]`.

* `CoordConv` : 2D convolution module using extra coordinate information.

* `CoodConvNet` : This module improves 2D convolutions inside a convnet by processing extra coordinate information. It adds coordinate information to inputs of each 2D convolution module (`torch.nn.Conv2d`). Convnet model must contain single **Sequential** container (`torch.nn.modules.container.Sequential`).

## Environment

* Python version : 2.7
* PyTorch version : 0.4.0
* Torchvision version : 0.2.1
