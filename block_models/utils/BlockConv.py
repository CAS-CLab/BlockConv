from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import numpy as np

class BlockConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, block_size, type=0, stride=1, padding=0, 
                dilation=1, groups=1, bias=True, padding_mode='constant' ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BlockConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
             False, _pair(0), groups, bias, 'zeros' if padding_mode=='constant' else padding_mode)

        self.padding_mode = padding_mode
        self.block_size = _pair(block_size)
        self.set_type(type)

    def set_type(self, type):
        self.type = type
        if type == 0:
            self.get_block_size_fn = self._type_F_block_size
        elif type == 1:
            self.get_block_size_fn = self._type_H_block_size
        else:
            raise ValueError('Only support type is 0 or 1, but got {}'.format(type))

    def _type_H_block_size(self, h, w):
        return ((int)(h / self.block_size[0]) if h % self.block_size[0] == 0 else h,
                            (int)(w / self.block_size[1]) if w % self.block_size[1] == 0 else w)
        
    def _type_F_block_size(self, h, w):
        return self.block_size

    def forward(self, input):
        batch_size, channel, i_h, i_w = input.shape
        
        block_size = self.get_block_size_fn(i_h, i_w)

        b_h, b_w = min(block_size[0], i_h), min(block_size[1], i_w)
              
        if b_h == 0:
            b_h = i_h
        if b_w == 0:
            b_w = i_w
        
        if b_h == i_h and b_w == i_w:
            expended_padding = (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
            if self.stride[0] == 2:
                return F.max_pool2d(F.conv2d(F.pad(input, expended_padding, mode=self.padding_mode),
                                                     self.weight, self.bias, 1, _pair(0), self.dilation, self.groups), 2)
            else:
                return F.conv2d(F.pad(input, expended_padding, mode=self.padding_mode), 
                                                self.weight, self.bias, 1, _pair(0), self.dilation, self.groups)

        block_num_h = int(np.ceil(i_h / b_h))
        block_num_w = int(np.ceil(i_w / b_w))

        new_input = input.view((batch_size, channel, block_num_h, b_h, block_num_w, b_w))

        new_input = new_input.permute(0, 2, 4, 1, 3, 5).contiguous()

        new_input = new_input.view((batch_size*block_num_h*block_num_w, channel, b_h, b_w))

        expended_padding = (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
        output =  F.conv2d(F.pad(new_input, expended_padding, mode=self.padding_mode),
                                         self.weight, self.bias, 1, _pair(0), self.dilation, self.groups)

        output = output.view((batch_size, block_num_h, block_num_w, self.out_channels, b_h, b_w))
        output = output.permute(0, 3, 1, 4, 2, 5).contiguous()
        output = output.view((batch_size, self.out_channels, i_h, i_w))

        if self.stride[0] == 2:
            output = F.max_pool2d(output, 2)

        return  output

    def extra_repr(self):
        s = super().extra_repr()
        s += ', block_size={}, type={}'.format(self.block_size, self.type)
        return s

if __name__ == '__main__':
    bconv = BlockConv2d(3, 32, 3, 2, 0, stride=1, padding=0, 
                dilation=1, groups=1, bias=True, padding_mode='constant')
    print(bconv)