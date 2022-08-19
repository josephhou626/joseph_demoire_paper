import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple

## Supervised Attention Module
# paper : Multi-Stage Progressive Image Restoration
# source code : https://github.com/swz30/MPRNet
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class SAM(nn.Module):
    def __init__(self, n_channel, kernel_size):
        super(SAM, self).__init__()
        self.conv1 = conv(n_channel, n_channel, kernel_size)
        self.conv2 = conv(n_channel, 3, kernel_size) 
        self.conv3 = conv(3, n_channel, kernel_size)

    def forward(self, x, moire_img):
        x1 = self.conv1(x) 


        out_img = self.conv2(x) + moire_img
        

        x2 = torch.sigmoid(self.conv3(out_img))
        x1 = x1*x2
        x1 = x1+x
        return x1, out_img


## Supervised Attention Module
# paper : BANet: Blur-aware Attention Networks for Dynamic Scene Deblurring
# source code : https://github.com/pp00704831/BANet
class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(SPBlock, self).__init__()
        midplanes = int(outplanes//2)


        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1_h = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_1_w = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)

        self.pool_3_h = nn.AdaptiveAvgPool2d((None, 3))
        self.pool_3_w = nn.AdaptiveAvgPool2d((3, None))
        self.conv_3_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_3_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_5_h = nn.AdaptiveAvgPool2d((None, 5))
        self.pool_5_w = nn.AdaptiveAvgPool2d((5, None))
        self.conv_5_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_5_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_7_h = nn.AdaptiveAvgPool2d((None, 7))
        self.pool_7_w = nn.AdaptiveAvgPool2d((7, None))
        self.conv_7_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_7_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.fuse_conv = nn.Conv2d(midplanes * 4, midplanes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

        self.conv_final = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)

        self.mask_conv_1 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu  = nn.ReLU(inplace=False)
        self.mask_conv_2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)


    def forward(self, x):
        _, _, h, w = x.size()
        x_1_h = self.pool_1_h(x)
        x_1_h = self.conv_1_h(x_1_h)
        x_1_h = x_1_h.expand(-1, -1, h, w)

        x_1_w = self.pool_1_w(x)
        x_1_w = self.conv_1_w(x_1_w)
        x_1_w = x_1_w.expand(-1, -1, h, w)

        x_3_h = self.pool_3_h(x)
        x_3_h = self.conv_3_h(x_3_h)
        x_3_h = F.interpolate(x_3_h, (h, w))

        x_3_w = self.pool_3_w(x)
        x_3_w = self.conv_3_w(x_3_w)
        x_3_w = F.interpolate(x_3_w, (h, w))

        x_5_h = self.pool_5_h(x)
        x_5_h = self.conv_5_h(x_5_h)
        x_5_h = F.interpolate(x_5_h, (h, w))

        x_5_w = self.pool_5_w(x)
        x_5_w = self.conv_5_w(x_5_w)
        x_5_w = F.interpolate(x_5_w, (h, w))

        x_7_h = self.pool_7_h(x)
        x_7_h = self.conv_7_h(x_7_h)
        x_7_h = F.interpolate(x_7_h, (h, w))

        x_7_w = self.pool_7_w(x)
        x_7_w = self.conv_7_w(x_7_w)
        x_7_w = F.interpolate(x_7_w, (h, w))

        hx = self.relu(self.fuse_conv(torch.cat((x_1_h + x_1_w, x_3_h + x_3_w, x_5_h + x_5_w, x_7_h + x_7_w), dim = 1)))
        mask_1 = self.conv_final(hx).sigmoid()
        out1 = x * mask_1

       
        hx = self.mask_relu(self.mask_conv_1(out1))
        mask_2 = self.mask_conv_2(hx).sigmoid()
        hx = out1 * mask_2

        return hx

#################################################################################
# paper : Wavelet-based and Dual-branch Neural Network for Demoireing
# source code : https://github.com/laulampaul/WDNet_demoire
class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
     
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


class Conv2d(_ConvNd): 

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

      
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

#################################################################################

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Residual block
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


# Dilated-Dense Attention
class DDA(nn.Module):

    def __init__(self, inplanes = 64 , bias=True,delia=1):
        super(DDA, self).__init__()

        self.deli1 = nn.Sequential(
            Conv2d(inplanes, inplanes , 3, stride=1, dilation=1),
            nn.ReLU(inplace=True),
        )


        self.deli2 = nn.Sequential(
            Conv2d(inplanes*2, inplanes , 3, stride=1, dilation=2),
            nn.ReLU(inplace=True),
        )



        self.deli3 = nn.Sequential(
            Conv2d(inplanes*3, inplanes , 3, stride=1, dilation=3),
            nn.ReLU(inplace=True),
        )



        self.deli4 = nn.Sequential(
            Conv2d(inplanes*4, inplanes , 3, stride=1, dilation= 2 ),
            nn.ReLU(inplace=True),
        )


        self.deli5 = nn.Sequential(
            Conv2d(inplanes*5, inplanes , 3, stride=1, dilation= 1 ),
            nn.ReLU(inplace=True),
        )


        self.cm1 = nn.Conv2d(inplanes*6, inplanes, kernel_size=(3, 3), stride = 1 , padding = 1  ,bias=False)

        self.cm2 = nn.Conv2d(inplanes, inplanes, kernel_size=(3, 3), stride = 1 , padding = 1  ,bias=False)

        self.seblock = SEBlock(inplanes)

    def forward(self, x):


        x1 = self.deli1(x)
        x2 = self.deli2(torch.cat((x, x1), 1))
        x3 = self.deli3(torch.cat((x, x1, x2), 1))
        x4 = self.deli4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.deli5(torch.cat((x, x1, x2, x3, x4), 1))


        DB_out = torch.cat((x, x1, x2, x3, x4 , x5), 1)

        cm1_out = self.cm1(DB_out)
 
        se_out = self.seblock(cm1_out)

        cm2_out = self.cm2(se_out)

        final_out = cm2_out + x 


        return final_out

# SE block
class SEBlock(nn.Module):
  def __init__(self, input_dim):
    super(SEBlock, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(input_dim, input_dim // 4),
      nn.ReLU(inplace=True),
      nn.Linear(input_dim // 4, input_dim),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y

class AFF_SE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF_SE, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            SEBlock(out_channel),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        x2 = self.conv(x)

        return x2



## Demoiréing Multi-Scale Feature Interaction (DMSFNI)
class DMSFI1(nn.Module):
    def __init__(self , base_channel , se_base_channel):
        super(DMSFI1 , self).__init__()

        self.branch1_rb1 = ResBlock(base_channel, base_channel)
        self.branch2_rb1 = ResBlock(base_channel*2, base_channel*2)
        self.branch3_rb1 = ResBlock(base_channel*4, base_channel*4)


        self.bt1_2 = nn.Conv2d(base_channel, base_channel*2, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 64 -> 128
        self.bt2_1 = nn.Conv2d(base_channel*2, base_channel, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 128 -> 64
        self.bt2_3 = nn.Conv2d(base_channel*2, base_channel*4, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 128 -> 256 
        self.bt3_2 = nn.Conv2d(base_channel*4, base_channel*2, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 256 -> 64

        self.bt1_se = SEBlock(base_channel)
        self.bt2_se = SEBlock(base_channel*2)
        self.bt3_se = SEBlock(base_channel*4)

        self.branch1_conv2 = ResBlock(base_channel , base_channel) 
        self.branch2_conv2 = ResBlock(base_channel*2 , base_channel*2)
        self.branch3_conv2 = ResBlock(base_channel*4 , base_channel*4)


        self.fusion_se = AFF_SE(base_channel * 7, se_base_channel ) 

    def forward(self, x1, x2 ,x4):


        branch1_out = self.branch1_rb1(x1)
        branch2_out = self.branch2_rb1(x2)
        branch3_out = self.branch3_rb1(x4)

        #middle fusion:
        branch1_2 = self.bt1_2(branch1_out)
        branch1_2 = F.interpolate(branch1_2, scale_factor=0.5)

        branch2_1 = self.bt2_1(branch2_out)
        branch2_1 = F.interpolate(branch2_1, scale_factor= 2 )
        
        branch2_3 = self.bt2_3(branch2_out)
        branch2_3 = F.interpolate(branch2_3, scale_factor=0.5)


        branch3_2 = self.bt3_2(branch3_out)
        branch3_2 = F.interpolate(branch3_2, scale_factor= 2 )

 
        branch1_out = self.branch1_conv2((branch1_out + branch2_1)) 
        branch1_out = self.bt1_se(branch1_out)
        branch2_out = self.branch2_conv2((branch2_out + branch1_2 + branch3_2)) 
        branch2_out = self.bt2_se(branch2_out)
        branch3_out = self.branch3_conv2((branch3_out + branch2_3)) 
        branch3_out = self.bt3_se(branch3_out)


        #final fusion:
        branch2_out = F.interpolate(branch2_out, scale_factor=2)
        branch3_out = F.interpolate(branch3_out, scale_factor= 4 )


        output  = self.fusion_se(branch1_out , branch2_out , branch3_out)

        output = output + x1

        return output


class DMSFI2(nn.Module):
    def __init__(self , base_channel , se_base_channel):
        super(DMSFI2 , self).__init__()


        self.branch1_rb1 = ResBlock(base_channel, base_channel)
        self.branch2_rb1 = ResBlock(base_channel*2, base_channel*2)
        self.branch3_rb1 = ResBlock(base_channel*4, base_channel*4)
        

        self.bt1_2 = nn.Conv2d(base_channel, base_channel*2, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 64 -> 128
        self.bt2_1 = nn.Conv2d(base_channel*2, base_channel, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 128 -> 64
        self.bt2_3 = nn.Conv2d(base_channel*2, base_channel*4, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 128 -> 256 
        self.bt3_2 = nn.Conv2d(base_channel*4, base_channel*2, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 256 -> 64

        self.bt1_se = SEBlock(base_channel)
        self.bt2_se = SEBlock(base_channel*2)
        self.bt3_se = SEBlock(base_channel*4)


        self.branch1_conv2 = ResBlock(base_channel , base_channel) 
        self.branch2_conv2 = ResBlock(base_channel*2 , base_channel*2)
        self.branch3_conv2 = ResBlock(base_channel*4 , base_channel*4)

        self.fusion_se = AFF_SE(base_channel * 7, se_base_channel ) 

    def forward(self, x1, x2 ,x4):


        branch1_out = self.branch1_rb1(x1)
        branch2_out = self.branch2_rb1(x2)
        branch3_out = self.branch3_rb1(x4)


        #middle fusion:
        branch1_2 = self.bt1_2(branch1_out)
        branch1_2 = F.interpolate(branch1_2, scale_factor=0.5)

        branch2_1 = self.bt2_1(branch2_out)
        branch2_1 = F.interpolate(branch2_1, scale_factor= 2 )
        
        branch2_3 = self.bt2_3(branch2_out)
        branch2_3 = F.interpolate(branch2_3, scale_factor=0.5)


        branch3_2 = self.bt3_2(branch3_out)
        branch3_2 = F.interpolate(branch3_2, scale_factor= 2 )


        branch1_out = self.branch1_conv2((branch1_out + branch2_1)) 
        branch1_out = self.bt1_se(branch1_out) 
        branch2_out = self.branch2_conv2((branch2_out + branch1_2 + branch3_2)) 
        branch2_out = self.bt2_se(branch2_out) 
        branch3_out = self.branch3_conv2((branch3_out + branch2_3))  
        branch3_out = self.bt3_se(branch3_out) 


        ##final fusion:
        branch1_out = F.interpolate(branch1_out, scale_factor=0.5)
        branch3_out = F.interpolate(branch3_out, scale_factor= 2 )


        output  = self.fusion_se(branch1_out , branch2_out , branch3_out)

        output = output + x2

        return output


class DMSFI3(nn.Module):
    def __init__(self , base_channel , se_base_channel):
        super(DMSFI3 , self).__init__()


        self.branch1_rb1 = ResBlock(base_channel, base_channel)
        self.branch2_rb1 = ResBlock(base_channel*2, base_channel*2)
        self.branch3_rb1 = ResBlock(base_channel*4, base_channel*4)
        

        self.bt1_2 = nn.Conv2d(base_channel, base_channel*2, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 64 -> 128
        self.bt2_1 = nn.Conv2d(base_channel*2, base_channel, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 128 -> 64
        self.bt2_3 = nn.Conv2d(base_channel*2, base_channel*4, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 128 -> 256 
        self.bt3_2 = nn.Conv2d(base_channel*4, base_channel*2, kernel_size=(1, 1) , stride = 1 , padding = 0  ,bias=False) # channel 256 -> 64

 
        self.bt1_se = SEBlock(base_channel)
        self.bt2_se = SEBlock(base_channel*2)
        self.bt3_se = SEBlock(base_channel*4)


        self.branch1_conv2 = ResBlock(base_channel , base_channel) 
        self.branch2_conv2 = ResBlock(base_channel*2 , base_channel*2)
        self.branch3_conv2 = ResBlock(base_channel*4 , base_channel*4)
 
        self.fusion_se = AFF_SE(base_channel * 7, se_base_channel ) 

    def forward(self, x1, x2 ,x4):


        branch1_out = self.branch1_rb1(x1)
        branch2_out = self.branch2_rb1(x2)
        branch3_out = self.branch3_rb1(x4)


        #middle fusion
        branch1_2 = self.bt1_2(branch1_out)
        branch1_2 = F.interpolate(branch1_2, scale_factor=0.5)

        branch2_1 = self.bt2_1(branch2_out)
        branch2_1 = F.interpolate(branch2_1, scale_factor= 2 )
        
        branch2_3 = self.bt2_3(branch2_out)
        branch2_3 = F.interpolate(branch2_3, scale_factor=0.5)


        branch3_2 = self.bt3_2(branch3_out)
        branch3_2 = F.interpolate(branch3_2, scale_factor= 2 )


        branch1_out = self.branch1_conv2((branch1_out + branch2_1)) 
        branch1_out = self.bt1_se(branch1_out) 
        branch2_out = self.branch2_conv2((branch2_out + branch1_2 + branch3_2)) 
        branch2_out = self.bt2_se(branch2_out) 
        branch3_out = self.branch3_conv2((branch3_out + branch2_3)) 
        branch3_out = self.bt3_se(branch3_out) 

        #final fusion:
        branch1_out = F.interpolate(branch1_out, scale_factor= 0.25)
        branch2_out = F.interpolate(branch2_out, scale_factor= 0.5 )

        output  = self.fusion_se(branch1_out , branch2_out , branch3_out)

        output = output + x4

        return output


class DMSFN_plus(nn.Module):
    def __init__(self):
        super(DMSFN_plus, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            DDA(base_channel ),
            DDA(base_channel*2),
            DDA(base_channel*4),
        ])

        self.Encoder2 = nn.ModuleList([
            DDA(base_channel ),
            DDA(base_channel*2),
            DDA(base_channel*4),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])


        self.Decoder = nn.ModuleList([
            DDA(base_channel*4 ),
            DDA(base_channel*2),
            DDA(base_channel),
        ])

        self.SPB = nn.ModuleList([
            SPBlock(base_channel*4, base_channel*4),
            SPBlock(base_channel*2,base_channel*2),
            SPBlock(base_channel,base_channel),
        ])


        self.Decoder2 = nn.ModuleList([
            DDA(base_channel*4 ),
            DDA(base_channel*2),
            DDA(base_channel),
        ])


        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.SAM = nn.ModuleList([
            SAM(base_channel * 4, 1),
            SAM(base_channel * 2, 1),
        ])


        self.AFFs = nn.ModuleList([
            DMSFI1(base_channel , base_channel*1),
            DMSFI2(base_channel , base_channel*2),
            DMSFI3(base_channel , base_channel*4)
        ])



    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)  
        x_4 = F.interpolate(x_2, scale_factor=0.5) 

        outputs = list() 

        ###########Encoder#############
        #branch1:
        x_ = self.feat_extract[0](x) # channel: 3-> 32 
        res1 = self.Encoder[0](x_) 
        res1 = self.Encoder2[0](res1)

        #branch2:
        z = self.feat_extract[1](res1) 
        res2 = self.Encoder[1](z) 
        res2 = self.Encoder2[1](res2)

        #branch3:
        z = self.feat_extract[2](res2) 
        z = self.Encoder[2](z) 
        z = self.Encoder2[2](z)

        res1_out = self.AFFs[0](res1, res2, z) # DMSFNI1
        res2_out = self.AFFs[1](res1, res2, z) # DMSFNI2
        res3_out = self.AFFs[2](res1,res2,z) # DMSFNI3

        ###########Decoder#############
        #branch3:
        z = self.Decoder[0](res3_out) 
        z = self.SPB[0](z) 
        z = self.Decoder2[0](z) 
        z , z_ = self.SAM[0](z,x_4)
        z = self.feat_extract[3](z) 
        outputs.append(z_)
        
        #branch2:
        z = torch.cat([z, res2_out], dim=1)
        z = self.Convs[0](z) 
        z = self.Decoder[1](z) 
        z = self.SPB[1](z) 
        z = self.Decoder2[1](z) 
        z , z_ = self.SAM[1](z,x_2)
        z = self.feat_extract[4](z) 
        outputs.append(z_)

        #branch1:
        z = torch.cat([z, res1_out], dim=1)
        z = self.Convs[1](z) 
        z = self.Decoder[2](z) 
        z = self.SPB[2](z) 
        z = self.Decoder2[2](z) 


        #final:
        z = self.feat_extract[5](z) 

        outputs.append(z+x) 

        return outputs

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMSFN_plus().to(device)
    summary(model , (3,256,256))  
