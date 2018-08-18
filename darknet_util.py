import os,sys
import torch.nn

class YoloLayer(torch.nn.Module):
    def __init__(self, original_img_size, anchors):
        super(YoloLayer, self).__init__()
        self.original_img_size = original_img_size
        self.anchors = anchors


    def forward(self, x):
        input_dim = x.shape
        batch_size = input_dim[0]
        channels = input_dim[1]
        input_rows = input_dim[2]
        input_cols = input_dim[3]
