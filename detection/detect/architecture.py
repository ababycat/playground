import torch
from torch import nn
import torchvision as vision

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
#         self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = vision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True))

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
#         self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
#         self.dec0 = ConvRelu(num_filters, num_filters)
#         self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # center = self.center(self.conv6(conv5))
        # dec5 = self.dec5(torch.cat([center, conv5], 1))
        
        out = self.conv6(conv5)
        return out

#         dec4 = self.dec4(torch.cat([dec5, conv4], 1))
#         dec3 = self.dec3(torch.cat([dec4, conv3], 1))
#         dec2 = self.dec2(torch.cat([dec3, conv2], 1))
#         dec1 = self.dec1(dec2)
#         dec0 = self.dec0(dec1)

#         if self.num_classes > 1:
#             x_out = F.log_softmax(self.final(dec0), dim=1)
#         else:
#             x_out = self.final(dec0)
        # return dec5


class RNN_Decoder(nn.Module):
    def __init__(self, samples_size, input_size, hidden_size, linear_output_size, decode_times):
        super().__init__()
        self.samples_size = samples_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.times = decode_times
        self.output_size = linear_output_size
    
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
#         self.leaky1 = nn.LeakyReLU()
#         self.bn1 = nn.BatchNorm1d(self.hidden_size)
#         self.dropout1 = nn.Dropout(0.1)
#         self.linear1 = nn.Linear(self.hidden_size, 256)
        
#         self.leaky2 = nn.LeakyReLU()
#         self.bn2 = nn.BatchNorm1d(256)
#         self.dropout2 = nn.Dropout(0.1)
#         self.linear2 = nn.Linear(256, self.output_size)
        self.leaky1 = nn.LeakyReLU()
#         self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(self.hidden_size, linear_output_size)

    def set_decode_times(self, times):
        self.times = times

    def forward(self, x, hidden):
        """x : BSx256x40x40"""
        hx, cx = hidden
        bs, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        # x: (BSx40x40)x256
        x = x.reshape(-1, x.shape[-1])
        
        output = []
        for i in range(self.times):
            hx, cx = self.lstm(x, [hx, cx])
            output.append(hx.unsqueeze(0))
        
        # linear_input: timesx(BSx40x40)xhidden_size
        linear_input = torch.cat(output, dim=0)
        # (timesx(BSx40x40))xhidden_size
        x = linear_input.reshape(-1, linear_input.shape[-1])
        x = self.dropout1(x)
        x = self.leaky1(x)
#         x = self.bn1(x)
        x = self.linear1(x)
#         x = self.dropout2(x)
#         x = self.leaky2(x)
#         x = self.bn2(x)
#         x = self.linear2(x)
        # TODO: x: (timesx(BSx40x40))x85 -> timesxBSx40x40x85 -> BSx40x40xtimesx85
        x = x.reshape(self.times, bs, h, w, -1)
        # BSx40x40xtimesx85
        x = x.permute(1, 2, 3, 0, 4)
#         x = x.reshape(bs, h*w*self.times, -1)
        return x
    
    def init_hidden_state(self, sample_size):
        self.samples_size = sample_size
        return torch.zeros(self.samples_size, self.hidden_size), torch.zeros(self.samples_size, self.hidden_size)


class DetectNet(nn.Module):
    def __init__(self, rpn_model, detect_model, scale=1):
        super().__init__()
        self.rpn = rpn_model
        self.detect_model = detect_model
        self.scale = scale
        
    def forward(self, x, hidden):
        rpn_output = self.rpn(x)
        detect_output = self.detect_model(rpn_output*self.scale, hidden)
        return detect_output
    
    def init_rnn_state(self, sample_size):
        return self.detect_model.init_hidden_state(sample_size)
