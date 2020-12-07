import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from torch.autograd import Variable


from torch import Tensor, Size
from torch.jit.annotations import List, Optional, Tuple


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        n: Optional[int] = None,
    ):
        # n=None for backward-compatibility
        if n is not None:
            warnings.warn(
                "`n` argument is deprecated and has been renamed `num_features`",
                DeprecationWarning,
            )
            num_features = n
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution.


    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        freeze_bn=False,
    ):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        #         self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.bn = FrozenBatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        freeze_bn=False,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        #         self.bn = FrozenBatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=224, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        # self.p6_td = DepthwiseConvBlock(feature_size, feature_size)

        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        #         self.p7_out = DepthwiseConvBlock(feature_size, feature_size)

        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 3), requires_grad=True)
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 3), requires_grad=True)
        self.w2_relu = nn.ReLU()

    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x = inputs

        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon

        # p7_td = p7_x
        # p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, scale_factor=2))
        p6_td = p6_x
        p5_td = self.p5_td(
            w1[0, 0] * p5_x + w1[1, 0] * F.interpolate(p6_td, scale_factor=2)
        )
        # p5_td = p5_x
        p4_td = self.p4_td(
            w1[0, 1] * p4_x + w1[1, 1] * F.interpolate(p5_td, scale_factor=1)
        )
        p3_td = self.p3_td(
            w1[0, 2] * p3_x + w1[1, 2] * F.interpolate(p4_td, scale_factor=2)
        )

        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(
            w2[0, 0] * p4_x
            + w2[1, 0] * p4_td
            + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out)
        )
        p5_out = self.p5_out(
            w2[0, 1] * p5_x
            + w2[1, 1] * p5_td
            + w2[2, 1] * nn.Upsample(scale_factor=1)(p4_out)
        )
        p6_out = self.p6_out(
            w2[0, 2] * p6_x
            + w2[1, 2] * p6_td
            + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out)
        )
        # p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * nn.Upsample(scale_factor=0.5)(p6_out))

        return [p3_out, p4_out, p5_out, p6_out]


class BiFPN(nn.Module):
    def __init__(self, size, feature_size=224, num_layers=1, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.p3 = nn.Conv2d(
            size[0], feature_size, kernel_size=1, stride=1, padding=0
        )
        self.p4 = nn.Conv2d(
            size[1], feature_size, kernel_size=1, stride=1, padding=0
        )
        self.p5 = nn.Conv2d(
            size[2], feature_size, kernel_size=1, stride=1, padding=0
        )
        self.p6 = ConvBlock(
            feature_size, feature_size, kernel_size=3, stride=2, padding=1
        )

        # p6 is obtained via a 3x3 stride-2 conv on C5
        #         self.p6 = nn.Conv2d(size[2], feature_size, kernel_size=3, stride=2, padding=1)

        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        #         self.p7 = ConvBlock(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)

    def forward(self, inputs):
        c3, c4, c5 = inputs["0"], inputs["1"], inputs["2"]
        # print(c3.shape,c4.shape,c5.shape)
        # print(c3.shape[2])

        # align for input size
        if (c3.shape[2] / 2) % 2 != 0:
            c3_align = nn.ZeroPad2d((0, 0, 1, 1))
            c4_align = nn.ZeroPad2d((0, 0, 1, 0))
            c5_align = nn.ZeroPad2d((0, 0, 1, 0))
            c3 = c3_align(c3)
            c4 = c4_align(c4)
            c5 = c5_align(c5)

        elif (c3.shape[3] / 2) % 2 != 0:
            c3_align = nn.ZeroPad2d((1, 1, 0, 0))
            c4_align = nn.ZeroPad2d((1, 0, 0, 0))
            c5_align = nn.ZeroPad2d((1, 0, 0, 0))
            c3 = c3_align(c3)
            c4 = c4_align(c4)
            c5 = c5_align(c5)

        # Calculate the input column of BiFPN
        p3_x = self.p3(c3)
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(p5_x)
        # print(p3_x.shape,p4_x.shape,p5_x.shape,p6_x.shape)

        features = [p3_x, p4_x, p5_x, p6_x]
        features_out = self.bifpn(features)

        out = OrderedDict()
        out["0"] = features_out[0]
        out["1"] = features_out[1]
        out["2"] = features_out[2]
        out["3"] = features_out[3]

        return out
