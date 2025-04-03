import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d, DeformConv2d
from torch.nn.init import constant_, xavier_uniform_
import math

class DCNv4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, offset_mask, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, group_channels, offset_scale, im2col_step, remove_center):
        # This function should implement the forward pass of the deformable convolution.
        # The actual implementation details are not provided here.
        # For demonstration purposes, this is a placeholder.
        ctx.save_for_backward(input, offset_mask)
        # Placeholder for actual deformable convolution operation
        output = input  # Replace with actual implementation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # This function should implement the backward pass of the deformable convolution.
        # The actual implementation details are not provided here.
        # For demonstration purposes, this is a placeholder.
        input, offset_mask = ctx.saved_tensors
        grad_input = grad_output  # Replace with actual implementation
        grad_offset_mask = grad_output  # Replace with actual implementation
        return grad_input, grad_offset_mask, None, None, None, None, None, None, None, None, None, None, None, None

# class CenterFeatureScaleModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Initialization of module components if needed
#
#     def forward(self, x, weight, bias):
#         # Implement the forward pass for center feature scaling
#         # Placeholder implementation
#         return x * weight + bias


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale

class DCNv4(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            dw_kernel_size=None,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs):
        """
        DCNv4 Module
        :param channels: Number of input channels
        :param kernel_size: Size of the convolution kernel
        :param stride: Stride for the convolution
        :param pad: Padding for the convolution
        :param dilation: Dilation rate for the convolution
        :param group: Number of groups for grouped convolution
        :param offset_scale: Scale factor for the offset
        :param dw_kernel_size: Kernel size for depth-wise convolution
        :param center_feature_scale: Whether to use center feature scaling
        :param remove_center: Whether to remove the center feature
        :param output_bias: Whether to use bias in the output projection
        :param without_pointwise: Whether to skip the point-wise convolution
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group

        # Ensure _d_per_group is a power of 2 for CUDA efficiency
        assert _d_per_group % 16 == 0

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.K = group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(channels, channels, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels)
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3) / 8) * 8))
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group,))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.)
        constant_(self.offset_mask.bias.data, 0.)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, input, shape=None):
        """
        Forward pass of the DCNv4 module
        :param input: Input tensor (N, L, C)
        :param shape: Shape of the input tensor (H, W)
        :return: Output tensor (N, H, W, C)
        """
        N,C,H,W = input.shape
        input = input.view(N, H*W, C)
        N, L, C = input.shape
        if shape is not None:
            H, W = shape
        else:
            H, W = int(L ** 0.5), int(L ** 0.5)


        x = input
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1)
        if self.dw_kernel_size:
            x = self.value_proj(x)

        x = x.reshape(N, H, W, -1)
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(input.view(N, H, W, C).permute(0, 3, 1, 2))
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        else:
            offset_mask_input = input
        offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)

        x_proj = x

        x = DCNv4Function.apply(
            x, offset_mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center
        )

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.view(N, L, -1)

        if not self.without_pointwise:
            x = self.output_proj(x)

        x = x.view(N, -1, H, W)
        return x


#调制可变形卷积v2
class DCNv2(nn.Module):
    r""" Deformable Convolution v4 (DCNv4)
    Args:

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DCNv2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
        self.mask_conv = nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        n, c, h, w = x.size()
        # offset = offset.view(n, -1, 2, h, w)
        # mask = mask.view(n, -1, h, w)
        offset = offset.view(n, 2 * self.kernel_size * self.kernel_size, h, w)  # Ensure offset shape is correct
        mask = mask.view(n, self.kernel_size * self.kernel_size, h, w)
        x = deform_conv2d(x, offset, self.conv.weight, self.conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)
        return x

class DWConv2(nn.Module):
    r""" Deformable Convolution v4 (DCNv4)
    Args:

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DWConv2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding= padding, dilation=dilation, bias=True)
        self.mask_conv = nn.Conv2d(in_channels, kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        n, c, h, w = x.size()
        # offset = offset.view(n, -1, 2, h, w)
        # mask = mask.view(n, -1, h, w)
        offset = offset.view(n, 2 * self.kernel_size[0] * self.kernel_size[1], h, w)  # Ensure offset shape is correct
        mask = mask.view(n, self.kernel_size[0] * self.kernel_size[1], h, w)
        x = deform_conv2d(x, offset, self.conv.weight, self.conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)
        return x

class AxialDCNv4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(AxialDCNv4, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # 水平方向卷积，使用可变形卷积
        self.conv_h_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding), dilation=(1, dilation))
        self.conv_h_weight = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding), dilation=(1, dilation))

        # 垂直方向卷积，使用可变形卷积
        self.conv_v_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0), dilation=(dilation, 1))
        self.conv_v_weight = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0), dilation=(dilation, 1))

    def forward(self, x):
        # 水平方向偏移和权重计算
        offset_h = self.conv_h_offset(x)
        offset_h = offset_h.view(offset_h.size(0), 2, self.kernel_size * self.kernel_size, offset_h.size(2), offset_h.size(3))
        offset_h = offset_h.permute(0, 1, 3, 4, 2).contiguous()
        offset_h = offset_h.view(offset_h.size(0), 2 * self.kernel_size * self.kernel_size, offset_h.size(3), offset_h.size(4))

        weight_h = self.conv_h_weight(x)

        # 垂直方向偏移和权重计算
        offset_v = self.conv_v_offset(x)
        offset_v = offset_v.view(offset_v.size(0), 2, self.kernel_size * self.kernel_size, offset_v.size(2), offset_v.size(3))
        offset_v = offset_v.permute(0, 1, 3, 4, 2).contiguous()
        offset_v = offset_v.view(offset_v.size(0), 2 * self.kernel_size * self.kernel_size, offset_v.size(3), offset_v.size(4))

        weight_v = self.conv_v_weight(x)

        # 水平方向的可变形卷积
        out_h = deform_conv2d(x, offset_h, weight_h, stride=(self.stride, self.stride), padding=(self.padding, self.padding), dilation=(self.dilation, self.dilation))

        # 垂直方向的可变形卷积
        out_v = deform_conv2d(out_h, offset_v, weight_v, stride=(self.stride, self.stride), padding=(self.padding, self.padding), dilation=(self.dilation, self.dilation))

        return out_v



class MDCNv4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(MDCNv4, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.kernel_size = kernel_size
        self.kernel_size = kernel_size[1]
        self.groups = groups

        # 初始化偏移和掩码卷积层
        self.offset_conv = nn.Conv2d(in_channels, 3 * kernel_size[1] * kernel_size[1], kernel_size=kernel_size[1], stride=stride, padding=padding, dilation=dilation, bias=True)
        # 初始化可变形卷积层
        self.deform_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[1], stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # 计算偏移量和掩码
        out = self.offset_conv(x)
        offset = out[:, :2 * self.kernel_size * self.kernel_size, :, :]
        mask = out[:, 2 * self.kernel_size * self.kernel_size:, :, :]
        mask = torch.sigmoid(mask)

        # 使用可变形卷积
        x = deform_conv2d(x, offset, self.deform_conv.weight, self.deform_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)

        return x

class MDCNv4_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(MDCNv4_1, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.kernel_size = kernel_size
        self.kernel_size = kernel_size[0]
        self.groups = groups

        # 初始化偏移和掩码卷积层
        self.offset_conv = nn.Conv2d(in_channels, 3 * kernel_size[0] * kernel_size[0], kernel_size=kernel_size[0], stride=stride, padding=padding, dilation=dilation, bias=True)
        # 初始化可变形卷积层
        self.deform_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[0], stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # 计算偏移量和掩码
        out = self.offset_conv(x)
        offset = out[:, :2 * self.kernel_size * self.kernel_size, :, :]
        mask = out[:, 2 * self.kernel_size * self.kernel_size:, :, :]
        mask = torch.sigmoid(mask)

        # 使用可变形卷积
        x = deform_conv2d(x, offset, self.deform_conv.weight, self.deform_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)

        return x


class AxialMDCNv4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(AxialMDCNv4, self).__init__()
        # 水平和垂直方向的调制可变形卷积
        self.axial_conv_h = MDCNv4(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(padding, padding), dilation=dilation)
        self.axial_conv_v = MDCNv4_1(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, padding), dilation=dilation)

    def forward(self, x):
        # 水平方向卷积
        out = self.axial_conv_h(x)
        # 垂直方向卷积
        out = self.axial_conv_v(out)
        return out

#可变形轴向卷积
class DAxialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True):
        super(DAxialConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.offset_conv_h = nn.Conv2d(in_channels, 2 * kernel_size, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, bias=True)
        self.mask_conv_h = nn.Conv2d(in_channels, kernel_size, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, bias=True)
        self.offset_conv_v = nn.Conv2d(in_channels, 2 * kernel_size, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, bias=True)
        self.mask_conv_v = nn.Conv2d(in_channels, kernel_size, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, bias=True)
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, bias=bias)
        self.conv_v = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, bias=bias)

    def forward(self, x):
        # Horizontal Deformable Convolution
        offset_h = self.offset_conv_h(x)
        mask_h = torch.sigmoid(self.mask_conv_h(x))
        n, c, h, w = x.size()
        offset_h = offset_h.view(n, -1, 2, h, w)
        mask_h = mask_h.view(n, -1, h, w)
        x_h = deform_conv2d(x, offset_h, self.conv_h.weight, self.conv_h.bias, stride=self.stride, padding=(0, self.padding), dilation=self.dilation, mask=mask_h)

        # Vertical Deformable Convolution
        offset_v = self.offset_conv_v(x)
        mask_v = torch.sigmoid(self.mask_conv_v(x))
        offset_v = offset_v.view(n, -1, 2, h, w)
        mask_v = mask_v.view(n, -1, h, w)
        x_v = deform_conv2d(x_h, offset_v, self.conv_v.weight, self.conv_v.bias, stride=self.stride, padding=(self.padding, 0), dilation=self.dilation, mask=mask_v)

        return x_v


#轴向卷积
class AxialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(AxialConv2d, self).__init__()
        #self.horizontal_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), bias=bias)
        #self.vertical_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), bias=bias)

        self.horizontal_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), bias=bias)
        self.vertical_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), bias=bias)
    def forward(self, x):
        x = self.horizontal_conv(x)
        x = self.vertical_conv(x)
        return x


#深度可分离轴向卷积
class AxialDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(AxialDWConv2d, self).__init__()
        self.depthwise_h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise_h = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.depthwise_v = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, groups=out_channels, bias=bias)
        self.pointwise_v = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias)

        # self.depthwise_v = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, groups=in_channels, bias=bias)
        # self.pointwise_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_h(x)
        x = self.pointwise_h(x)
        x = self.depthwise_v(x)
        x = self.pointwise_v(x)
        return x


#可选远近感知卷积
class OptionalNearFarAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding, stride=1, bias=False):
        super(OptionalNearFarAwareConv2d, self).__init__()
        self.num_kernels = len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
            for kernel_size in kernel_sizes
        ])
        self.attention_conv = nn.Conv2d(in_channels, self.num_kernels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        attention = self.attention_conv(x)
        attention = F.softmax(attention, dim=1)
        outputs = [conv(x) * attention[:, i:i+1, :, :] for i, conv in enumerate(self.convs)]
        return sum(outputs)

#可选远近感知卷积轴向卷积
class ONFAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding, stride=1, bias=False):
        super(ONFAConv2d, self).__init__()
        self.num_kernels = len(kernel_sizes)
        self.horizontal_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding), stride=stride, bias=bias)
            for kernel_size in kernel_sizes
        ])
        self.vertical_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0), stride=stride, bias=bias)
            for kernel_size in kernel_sizes
        ])
        self.attention_conv_h = nn.Conv2d(in_channels, self.num_kernels, kernel_size=(1, 1), padding=0, bias=True)
        self.attention_conv_v = nn.Conv2d(out_channels, self.num_kernels, kernel_size=(1, 1), padding=0, bias=True)

    def forward(self, x):
        attention_h = self.attention_conv_h(x)
        attention_h = F.softmax(attention_h, dim=1)
        horizontal_outputs = [conv(x) * attention_h[:, i:i+1, :, :] for i, conv in enumerate(self.horizontal_convs)]
        x_h = sum(horizontal_outputs)

        attention_v = self.attention_conv_v(x_h)
        attention_v = F.softmax(attention_v, dim=1)
        vertical_outputs = [conv(x_h) * attention_v[:, i:i+1, :, :] for i, conv in enumerate(self.vertical_convs)]
        x_v = sum(vertical_outputs)

        return x_v



class ONFAMDConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding, stride=1, bias=False):
        super(ONFAMDConv2d, self).__init__()
        self.num_kernels = len(kernel_sizes)
        self.horizontal_convs = nn.ModuleList([
            # AxialMDCNv4(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
            # AxialMDCNv4(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)
            DWConv2(in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2), stride=stride)
            for kernel_size in kernel_sizes
        ])
        self.vertical_convs = nn.ModuleList([
            # AxialMDCNv4(out_channels, out_channels, kernel_size, padding=padding, stride=stride)
            # AxialMDCNv4(out_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)
            DWConv2(out_channels, out_channels,(kernel_size, 1), padding=(kernel_size//2, 0), stride=stride)
            for kernel_size in kernel_sizes
        ])
        self.attention_conv_h = nn.Conv2d(in_channels, self.num_kernels, kernel_size=(1, 1), padding=0, bias=True)
        self.attention_conv_v = nn.Conv2d(out_channels, self.num_kernels, kernel_size=(1, 1), padding=0, bias=True)

    def forward(self, x):
        # Horizontal Convolution
        attention_h = self.attention_conv_h(x)
        attention_h = F.softmax(attention_h, dim=1)
        # for i,conv in enumerate(self.horizontal_convs):
        #     a = conv(x)
        #     b = attention_h[:, i:i+1, :, :]
        horizontal_outputs = [conv(x) * attention_h[:, i:i+1, :, :] for i, conv in enumerate(self.horizontal_convs)]
        x_h = sum(horizontal_outputs)

        # Vertical Convolution
        attention_v = self.attention_conv_v(x_h)
        attention_v = F.softmax(attention_v, dim=1)
        vertical_outputs = [conv(x_h) * attention_v[:, i:i+1, :, :] for i, conv in enumerate(self.vertical_convs)]
        x_v = sum(vertical_outputs)

        return x_v


class SingleONFAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding, stride=1, bias=False):
        super(SingleONFAConv2d, self).__init__()
        self.num_kernels = len(kernel_sizes)
        self.horizontal_convs = nn.ModuleList([
            AxialMDCNv4(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
            for kernel_size in kernel_sizes
        ])
        self.vertical_convs = nn.ModuleList([
            AxialMDCNv4(out_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
            for kernel_size in kernel_sizes
        ])
        self.attention_conv_h = nn.Conv2d(in_channels, self.num_kernels, kernel_size=(1, 1), padding=0, bias=True)
        self.attention_conv_v = nn.Conv2d(out_channels, self.num_kernels, kernel_size=(1, 1), padding=0, bias=True)

    def forward(self, x):
        # Horizontal Convolution
        attention_h = self.attention_conv_h(x)
        attention_h = torch.sigmoid(attention_h)
        horizontal_outputs = self.horizontal_convs(x) * attention_h
        x_h = sum(horizontal_outputs)

        # Vertical Convolution
        attention_v = self.attention_conv_v(x_h)
        attention_v = torch.sigmoid(attention_v, dim=1)
        vertical_outputs = self.vertical_convs(x_h) * attention_v
        x_v = sum(vertical_outputs)

        return x_v

class HeterogeneousConv_ori(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding, stride=1, bias=False):
        super(HeterogeneousConv, self).__init__()
        self.dcn_v4 = DCNv4(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.axial_convs = nn.ModuleList([
            AxialConv2d(in_channels, out_channels, kernel_size=ks, padding=ks//2, stride=stride, bias=bias)
            for ks in kernel_sizes
        ])
        self.optional_conv = OptionalNearFarAwareConv2d(in_channels, out_channels, kernel_sizes, padding=1, stride=stride, bias=bias)

    def forward(self, x):
        dcn_output = self.dcn_v4(x)
        axial_outputs = [conv(x) for conv in self.axial_convs]
        optional_output = self.optional_conv(x)
        output = dcn_output + optional_output + sum(axial_outputs)
        return output


class HeterogeneousConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding=1, stride=1, bias=False):
        super(HeterogeneousConv, self).__init__()

        # self.dcn_v4 = DCNv4(in_channels, kernel_size=3, padding=1, stride=stride, group=in_channels//16, offset_scale=2)
        self.dcn_v4 = DCNv2(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        #self.dcn_v4 = AxialMDCNv4(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        # self.dcn_v4 = AxialDCNv4(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.axial_convs = nn.ModuleList([
            AxialDWConv2d(in_channels, out_channels, kernel_size=ks, padding=ks // 2, stride=stride,bias=bias )
            for ks in kernel_sizes
        ])
        self.optional_conv = ONFAMDConv2d(in_channels, out_channels, kernel_sizes=[5, 9, 17], padding=1, stride=stride, bias=bias)
        # self.optional_conv = ONFAMDConv2d(in_channels, out_channels, kernel_sizes=[3, 7, 13], padding=1, stride=stride,
        #                                   bias=bias)

    def forward(self, x):
        dcn_output = self.dcn_v4(x)
        axial_outputs = [conv(x) for conv in self.axial_convs]
        optional_output = self.optional_conv(x)
        output = dcn_output + optional_output + sum(axial_outputs)
        return output


class HeterogeneousConv_v1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, kernel_sizes1, padding=1, stride=1, bias=False):
        super(HeterogeneousConv_v1, self).__init__()

        # self.dcn_v4 = DCNv4(in_channels, kernel_size=3, padding=1, stride=stride, group=in_channels//16, offset_scale=2)
        self.dcn_v4 = DCNv2(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        #self.dcn_v4 = AxialMDCNv4(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        # self.dcn_v4 = AxialDCNv4(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.axial_convs = nn.ModuleList([
            AxialDWConv2d(in_channels, out_channels, kernel_size=ks, padding=ks // 2, stride=stride,bias=bias )
            for ks in kernel_sizes
        ])
        self.optional_conv = ONFAMDConv2d(in_channels, out_channels, kernel_sizes=kernel_sizes1, padding=1, stride=stride, bias=bias)
        # self.optional_conv = ONFAMDConv2d(in_channels, out_channels, kernel_sizes=[3, 7, 13], padding=1, stride=stride,
        #                                   bias=bias)

    def forward(self, x):
        dcn_output = self.dcn_v4(x)
        axial_outputs = [conv(x) for conv in self.axial_convs]
        optional_output = self.optional_conv(x)
        output = dcn_output + optional_output + sum(axial_outputs)
        return output