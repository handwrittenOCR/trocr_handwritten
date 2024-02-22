from typing import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from trocr_handwritten.utils.pad import pad


# subclassing nn.Module and initialize neural network layers in __init__
# every nn.Module subclass implements the operations on input data in the forward method
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, activation):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=filter_size,
                stride=1,
                bias=False,
                padding="same",
            ),
            activation,
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=filter_size,
                stride=1,
                bias=False,
                padding="same",
            ),
            activation,
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_space_num,
        res_depth,
        featRoot,
        filter_size,
        pool_size,
        activation,
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.scale_space_num = scale_space_num
        self.res_depth = res_depth
        self.featRoot = featRoot
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.activation = activation

        # Calculate number of feature maps
        features = []
        features.append(self.featRoot)
        for layer in range(1, self.scale_space_num - 1):
            features.append(features[layer - 1] * pool_size)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, filter_size, activation))
            in_channels = feature

        # bottom part of the U-Net, features[-1] is the last feature
        self.bottleneck = DoubleConv(
            features[-1], features[-1] * 2, filter_size, activation
        )

        # Up part of UNET (consists the "up" part (ConvTranspose2d) and two following convs (DoubleConv))
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    # in_channels = feature*2 because we add the skip connection, i.e. 512 -> 1024
                    # out_channels = feature
                    feature * self.pool_size,
                    feature,
                    kernel_size=filter_size,
                    stride=2,
                )
            )

            self.ups.append(
                DoubleConv(feature * self.pool_size, feature, filter_size, activation)
            )

        # last step of the up part (64 -> 1 (in the paper they use 64 -> 2, but we have only 1 out channel))
        self.final_conv = nn.Conv2d(
            features[0], out_channels, kernel_size=4, stride=1, padding="same"
        )  # in the ARU-Net version a kernel size of 4 is used for the final conv of the U-Net

    def forward(self, x):
        skip_connections = []

        # run the "down" part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            max_pooling = pooling_same_padding(
                x,
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding="SAME",
                poolingType="MAX",
            )
            x = max_pooling(x)

        x = self.bottleneck(x)
        # reverse skip connections list
        skip_connections = skip_connections[::-1]

        # now we iterate over the ups with a step size of 2, because we going up and then double convs
        # up and double convs is both a own index in the list, i.e. : [up, doubleconv, up, doubleconv]
        # we only want the up index
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            # when the input shape is not the same as the output shape
            # for example 161x161 -> 80x80 -> 160x160
            if x.shape != skip_connection.shape:
                # so we do a resize
                # shape[2:] gets the height and width, skipping batch_size and amount of channels (tensor shape is [batch_size, channels, height, width])
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        x = self.final_conv(x)
        return x


############################### RU-NET ###############################


class RU_additionalConvs(nn.Module):
    def __init__(self, out_channels, res_depth, filter_size, activation):
        super(RU_additionalConvs, self).__init__()
        self.additionalConvs = nn.ModuleList()
        for aRes in range(0, res_depth):
            if aRes < res_depth - 1:  # no activation for last layer
                self.additionalConvs.append(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=filter_size,
                        stride=1,
                        bias=False,
                        padding="same",
                    )
                )
                self.additionalConvs.append(activation)
            else:
                self.additionalConvs.append(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=filter_size,
                        stride=1,
                        bias=False,
                        padding="same",
                    )
                ),

    def forward(self, x):
        for i in range(0, len(self.additionalConvs)):
            x = self.additionalConvs[i](x)

        return x


class RU_step(nn.Module):
    def __init__(self, in_channels, out_channels, res_depth, filter_size, activation):
        super(RU_step, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=filter_size,
            stride=1,
            bias=False,
            padding="same",
        )
        self.activation = activation

        self.additionalConvs = RU_additionalConvs(
            out_channels, res_depth, filter_size, activation
        )

    def forward(self, x):
        x = self.conv(x)
        orig_x = x
        x = self.activation(x)
        x = self.additionalConvs(x)
        x = orig_x + x
        result = self.activation(x)
        return result


class RU_deconv(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, activation):
        super(RU_deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=filter_size, stride=2
        )
        self.activation = activation

    def forward(self, x):
        return self.activation(self.deconv(x))


class RU_finalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RU_finalConv, self).__init__()
        self.final_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=1, padding="same"
        )

    def forward(self, x):
        return self.final_conv(x)


class RUNET(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_space_num,
        res_depth,
        featRoot,
        filter_size,
        pool_size,
        activation,
        final_conv_bool,
    ):
        super(RUNET, self).__init__()
        self.ru_ups = nn.ModuleList()
        self.ru_downs = nn.ModuleList()

        self.scale_space_num = scale_space_num
        self.res_depth = res_depth
        self.featRoot = featRoot
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.activation = activation

        # Calculate number of feature maps
        features = []
        features.append(self.featRoot)
        for layer in range(1, self.scale_space_num - 1):
            features.append(features[layer - 1] * pool_size)

        # Down part of RUNET
        for feature in features:
            self.ru_downs.append(
                RU_step(
                    in_channels,
                    feature,
                    self.res_depth,
                    self.filter_size,
                    self.activation,
                )
            )
            in_channels = feature

        # bottom part of the RU-Net, features[-1] is the last feature
        self.ru_bottleneck = RU_step(
            features[-1],
            features[-1] * self.pool_size,
            self.res_depth,
            self.filter_size,
            self.activation,
        )

        # Up part of RUNET (consists the "up" part (ConvTranspose2d) and two following convs (DoubleConv))
        for feature in reversed(features):
            self.ru_ups.append(
                # in_channels = feature*2 because we add the skip connection, i.e. 512 -> 1024
                # out_channels = feature
                RU_deconv(
                    feature * self.pool_size, feature, self.filter_size, self.activation
                )
            )

            self.ru_ups.append(
                RU_step(
                    feature * self.pool_size,
                    feature,
                    self.res_depth,
                    self.filter_size,
                    self.activation,
                )
            )

        # last step of the up part (64 -> 1 (in the paper they use 64 -> 2, but we have only 1 out channel))
        # If we use the RU-Net inside the ARU-Net, the final conv is done by the ARU-Net
        # If we only use the RU-Net, we need the final conv
        self.ru_final_conv = nn.Identity()
        if final_conv_bool:
            self.ru_final_conv = RU_finalConv(features[0], out_channels)

    def forward(self, x):
        skip_connections = []

        # run the "down" part
        for down in self.ru_downs:
            x = down(x)
            skip_connections.append(x)
            max_pooling = pooling_same_padding(
                x,
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding="SAME",
                poolingType="MAX",
            )
            x = max_pooling(x)

        x = self.ru_bottleneck(x)
        # reverse skip connections list
        skip_connections = skip_connections[::-1]

        # now we iterate over the ups with a step size of 2, because we going up and then double convs
        # up and double convs is both a own index in the list, i.e. : [up, convs, up, convs, ..]
        # we only want the "up" index
        for i in range(0, len(self.ru_ups), 2):
            x = self.ru_ups[i](x)  # deconvolution / up
            skip_connection = skip_connections[i // 2]

            # when the input shape is not the same as the output shape
            # for example 161x161 -> 80x80 -> 160x160
            if x.shape != skip_connection.shape:
                # so we do a resize
                # shape[2:] gets the height and width, skipping batch_size and amount of channels (tensor shape is [batch_size, channels, height, width])
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ru_ups[i + 1](concat_skip)

        # In the RU-Net for ARU-Net we don't use the final convolution.
        x = self.ru_final_conv(x)
        return x


############################### A-NET ###############################
class A_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(A_conv, self).__init__()
        self.conv = nn.Sequential(
            # in the original ARU-NET paper they use kernel_size of 4x4 in "attCNN" (Attention Network)
            nn.Conv2d(in_channels, out_channels, 4, 1, padding="same", bias=False),
        )

    def forward(self, x):
        return self.conv(x)


# This is based on "attCNN" of the ARU-Net (Tensorflow)
class ANET(nn.Module):
    # in_channels 1 or 3?
    def __init__(self, in_channels, activation):
        super(ANET, self).__init__()
        self.a_layers = nn.ModuleList()
        self.activation = activation

        self.a_layers.append(A_conv(in_channels, 12))
        # self.a_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.a_layers.append(A_conv(12, 16))
        # self.a_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.a_layers.append(A_conv(16, 32))
        # self.a_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.a_layers.append(A_conv(32, 1))

    def forward(self, x):
        counter = 1
        for layer in self.a_layers:
            x = layer(x)
            # after convolution we use activation function
            x = self.activation(x)
            # after last conv, there is no max pooling
            if counter <= 3:
                maxpooling = pooling_same_padding(
                    x, kernel_size=2, stride=2, padding="SAME", poolingType="MAX"
                )
                x = maxpooling(x)
            counter += 1

        return x


############################### ARU-NET ###############################
# The ARU-NET consists of the A-Net and RU-Net
class ARUNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        scale_space_num=6,
        res_depth=3,
        featRoot=8,
        filter_size=3,
        pool_size=2,
        activation=nn.ReLU(),
        num_scales=5,
    ):
        super(ARUNET, self).__init__()

        self.scale_space_num = scale_space_num
        self.res_depth = res_depth
        self.featRoot = featRoot
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.activation = activation
        self.num_scales = num_scales

        self.a_net = ANET(in_channels=in_channels, activation=self.activation)
        self.ru_net = RUNET(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_space_num=self.scale_space_num,
            res_depth=self.res_depth,
            featRoot=self.featRoot,
            filter_size=self.filter_size,
            pool_size=self.pool_size,
            activation=self.activation,
            final_conv_bool=False,
        )

        self.ru_net_upsample = nn.ModuleList()
        # we don't need to upscale the RU-part of the first image
        upSc = 1
        for sc in range(0, self.num_scales - 1):
            upSc = upSc * 2
            self.ru_net_upsample.append(
                nn.ConvTranspose2d(
                    in_channels=self.featRoot,
                    out_channels=self.featRoot,
                    kernel_size=upSc,
                    stride=upSc,
                )
            )

        self.a_net_upsample = nn.ModuleList()
        upSc = 8
        for sc in range(0, self.num_scales):
            self.a_net_upsample.append(
                nn.ConvTranspose2d(
                    in_channels=1, out_channels=1, kernel_size=upSc, stride=upSc
                )
            )
            upSc = upSc * 2

        self.a_net_softmax = nn.Softmax(dim=1)

        self.final_conv = nn.Conv2d(
            self.featRoot, out_channels, kernel_size=4, stride=1, padding="same"
        )

    def forward(self, x):
        # Average Pooling / Downscaling
        scaled_inputs = OrderedDict()
        scaled_inputs[0] = x
        for sc in range(1, self.num_scales):
            avgpool = pooling_same_padding(
                scaled_inputs[sc - 1],
                kernel_size=2,
                stride=2,
                padding="SAME",
                poolingType="AVG",
            )
            scaled_inputs[sc] = avgpool(scaled_inputs[sc - 1])

        # RU-Net Part
        ru_map = OrderedDict()
        ru_map[0] = self.ru_net(x)

        for sc in range(1, self.num_scales):
            out_S = self.ru_net(scaled_inputs[sc])
            # we don't need to upscale the RU-part of the first image
            # and the upsample layer starts at index 0 (not 1!)
            out = self.ru_net_upsample[sc - 1](out_S)
            ru_map[sc] = out

        # A-Net Part
        a_map = OrderedDict()
        for sc in range(0, self.num_scales):
            a_output = self.a_net(scaled_inputs[sc])
            a_map[sc] = self.a_net_upsample[sc](a_output)

        # Softmax
        # append all A-Nets together
        val = []
        for sc in range(0, self.num_scales):
            val.append(a_map[sc])
        a_maps_combined = torch.cat(val, dim=1)
        a_maps_softmax = self.a_net_softmax(a_maps_combined)

        # Split softmax tensor up again
        a_maps_split = torch.chunk(a_maps_softmax, self.num_scales, dim=1)

        # Point-wise multiplication of attention maps with feature maps of the RU-Net
        val = []
        for sc in range(0, self.num_scales):
            val.append(ru_map[sc] * a_maps_split[sc])

        maps_combined = sum(val)
        result = self.final_conv(maps_combined)
        return result


def pooling_same_padding(
    input, kernel_size, stride, padding="SAME", dilation=1, poolingType="AVG"
):
    """
    Calculates same padding for average and max pooling.
    Based on: https://github.com/Gasoonjia/Tensorflow-type-padding-with-pytorch-conv2d./blob/master/Conv2d_tensorflow.py#L104
    """

    def check_format(*argv):
        argv_format = []
        for i in range(len(argv)):
            if isinstance(argv[i], int):
                argv_format.append((argv[i], argv[i]))
            elif hasattr(argv[i], "__getitem__"):
                argv_format.append(tuple(argv[i]))
            else:
                raise TypeError(
                    "all input should be int or list-type, now is {}".format(argv[i])
                )

        return argv_format

    stride, dilation = check_format(stride, dilation)

    if padding == "SAME":
        padding = 0

        input_rows = input.size(2)
        filter_rows = kernel_size
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(
            0,
            (out_rows - 1) * stride[0]
            + (filter_rows - 1) * dilation[0]
            + 1
            - input_rows,
        )
        rows_odd = padding_rows % 2

        input_cols = input.size(3)
        filter_cols = kernel_size
        out_cols = (input_cols + stride[1] - 1) // stride[1]
        padding_cols = max(
            0,
            (out_cols - 1) * stride[1]
            + (filter_cols - 1) * dilation[1]
            + 1
            - input_cols,
        )
        cols_odd = padding_cols % 2

        input = pad(
            input,
            [
                padding_cols // 2,
                padding_cols // 2 + int(cols_odd),
                padding_rows // 2,
                padding_rows // 2 + int(rows_odd),
            ],
        )

    elif padding == "VALID":
        padding = 0

    elif isinstance(padding, int):
        raise ValueError(
            "Padding should be SAME, VALID or specific integer, but not {}.".format(
                padding
            )
        )
    if poolingType == "AVG":
        return nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        ).cuda()
    elif poolingType == "MAX":
        return nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        ).cuda()
    else:
        print("Please choose correct pooling operation!")
