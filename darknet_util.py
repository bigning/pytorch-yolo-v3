import os,sys
import torch.nn

class YoloLayer(torch.nn.Module):
    def __init__(self, original_img_size, anchors):
        '''
        original_img_size=[img_rows, img_cols]
        '''
        super(YoloLayer, self).__init__()
        self.original_img_size = original_img_size
        self.anchors = anchors

    def forward(self, x):
        if self.training:
            return x
        input_dim = x.shape
        batch_size = input_dim[0]
        channels = input_dim[1]
        input_rows = input_dim[2]
        input_cols = input_dim[3]
        num_anchors = len(self.anchors)

        stride = float(self.original_img_size[0]) / float(input_rows)

        # 5 = 4 + 1, 4 is the (x, y, w, h), 1 is the objectness score
        classes = channels / len(self.anchors) - 5
        #supose x is (batch, num_anchors*num_box_attr, rows, cols )
        # after view, -> (batch, num_anchors*num_box_attr, rows*cols)
        x = x.contiguous().view(batch_size, channels, input_rows * input_cols)
        # after transpose, -> (batch, rows * cols,num_anchors*num_box_attr)
        x = x.contiguous().transpose(1, 2)
        # after view -> (batch, rows*cols*num_anchors, num_box_attr)
        x = x.contiguous().view(
                batch_size,
                input_rows*input_cols*len(self.anchors),
                -1).contiguous()

        # bx = sigmoid(tx) + cx
        # by = sigmoid(ty) +cy
        # bw = pw * exp(tw)
        # bh = ph * exp(th)
        cx = torch.tensor([i for i in range(input_cols) for j in range(num_anchors)]).repeat(input_rows).to(torch.float)
        cy = torch.tensor([i for i in range(input_rows) for j in range(input_cols*num_anchors)]).to(torch.float)
        pw = torch.tensor([box[0] for box in self.anchors]).repeat(input_rows*input_cols).to(torch.float)
        ph = torch.tensor([box[1] for box in self.anchors]).repeat(input_rows*input_cols).to(torch.float)

        if x.is_cuda:
            cx = cx.cuda()
            cy = cy.cuda()
            pw = pw.cuda()
            ph = ph.cuda()
        x[:, :, 0] = (torch.sigmoid(x[:, :, 0]) + cx) * stride
        x[:, :, 1] = (torch.sigmoid(x[:, :, 1]) + cy) * stride
        x[:, :, 2] = torch.exp(x[:, :, 2]) * pw
        x[:, :, 3] = torch.exp(x[:, :, 3]) * ph

        # objectness and classes score
        x[:, :, 4:] = torch.sigmoid(x[:, :, 4:])
        
        # batch, rows*cols*num_anchors, num_box_attr
        return x
