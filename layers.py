import torch.nn as nn
import torch.nn.functional as F

def double_conv_relu(inC, outC):
    # Architecture: Conv 3x3 - ReLu - Conv 3x3 - ReLu
    return nn.Sequential(
        nn.Conv2d(inC, outC, kernel_size=3, stride=1, padding=1),
        nn.ReLu(),
        nn.nn.Conv2d(outC, outC, kernel_size=3, stride=1, padding=1),
        nn.ReLu()
    )