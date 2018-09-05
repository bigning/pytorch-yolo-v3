import torch
import math
import torch.nn
from darknet import DarknetYoloV3
from dataset import CocoDataset
import cv2
import torch.utils.data
import numpy as np
import sys, time,os, random
from main import detect_image

def compute_iou(bbox_priors, label):

    label = np.tile(np.asarray(label), (bbox_priors.shape[0], 1))
    xmin = np.maximum(bbox_priors[:, 0], label[:, 0])
    ymin = np.maximum(bbox_priors[:, 1], label[:, 1])
    xmax = np.minimum(bbox_priors[:, 0] + bbox_priors[:, 2], 
            label[:, 0] + label[:, 2])
    ymax = np.minimum(bbox_priors[:, 1] + bbox_priors[:, 3], 
            label[:, 1] + label[:, 3])
    intersection = np.maximum(xmax - xmin, 0) * np.maximum(ymax - ymin, 0)
    union = bbox_priors[:, 2] * bbox_priors[:, 3] + label[:, 2] * label[:, 3] - intersection

    ious = intersection/union

    return ious


class YoloV3Trainer(torch.nn.Module):
    def __init__(self, config_file, init_weight):
        super(YoloV3Trainer, self).__init__()
        self.yolov3 = DarknetYoloV3(config_file)
        
        self.yolov3.load_weight(init_weight, training=True)
        #self.yolov3.load_weight('./data/yolov3.weights', training=False)

        self.yolo_layer_size = []
        self.classes = 0
        self.anchors = []
        self.anchor_sizes = []

        self.get_bbox_priors()

        self.use_cuda = True

        self.bce_loss = torch.nn.BCELoss(reduction='elementwise_mean')
        self.mse_loss = torch.nn.MSELoss(reduction='elementwise_mean')
        if self.use_cuda:
            self.bce_loss = self.bce_loss.cuda()
            self.mse_loss = self.mse_loss.cuda()

    def forward(self, x):
        return self.yolov3(x)

    def compute_loss(self, yolo_outputs, labels):
        self.objectness_loss = 0.0
        self.coordinate_loss = 0.0
        self.classification_loss = 0.0

        batch_size = yolo_outputs[0].shape[0]
        yolo_layers = len(yolo_outputs)


        objectness_mask = []
        objectness_target = []
        coordinate_target = []
        coordinate_mask = []
        class_target = []
        class_mask = []

        for i in range(0, yolo_layers):
            objectness_mask.append(torch.ones([batch_size,
                self.anchors[i], self.yolo_layer_size[i], self.yolo_layer_size[i]]))
            objectness_target.append(torch.zeros([batch_size,
                self.anchors[i], self.yolo_layer_size[i], self.yolo_layer_size[i]]))

            coordinate_target.append(torch.zeros([batch_size,
                self.anchors[i] * 4, self.yolo_layer_size[i], self.yolo_layer_size[i]]))

            coordinate_mask.append(torch.zeros([batch_size,
                self.anchors[i] * 4, self.yolo_layer_size[i], self.yolo_layer_size[i]]))

            class_target.append(torch.zeros([batch_size,
                self.anchors[i] * self.classes, self.yolo_layer_size[i], self.yolo_layer_size[i]]))
            class_mask.append(torch.zeros([batch_size,
                self.anchors[i] * self.classes, self.yolo_layer_size[i], self.yolo_layer_size[i]]))
        ## set mask:
        # 1. initialize with 1
        # 2. for any label, if iou>0.5 
        #     a. if mask = 2, keep 2.  note:2 means this bbox is a maximum match to a taget
        #     b. else, mask = 0, 
        #     c. choose the maximum match bbox, set mask = 2
        # 3. mask = mask > 0

        ## set label:
        # 1. initialize with 1
        # 2. for any label
        #    if a bbox is has mamimux iou with label, set label = 1
        
        #debug_yolo

        obj_num = 0.0
        for i in range(0, batch_size):
            label = labels[i][0] ## shape (50, 5)
            
            for label_ind in range(0, label.shape[0]):
                if label[label_ind][0] < 0:
                    continue
                obj_num += 1

                ious = compute_iou(self.bbox_priors, label[label_ind][1:])
                for ious_ind, iou in enumerate(ious):
                    ## for bbox has >0.5 iou, set mask to 0, except the bbox
                    ## that's already maximumly matches a gt box
                    if iou < 0.5:
                        continue
                    yolo_layer_ind, anchor_ind, row, col = self.bbox_info[ious_ind]
                    if objectness_mask[yolo_layer_ind][i][anchor_ind][row][col] == 2:
                        continue
                    else:
                        objectness_mask[yolo_layer_ind][i][anchor_ind][row][col] = 0
                ## for bbox which has maximum iou to this taget box, set >0 mask
                max_ind = np.argmax(ious)
                max_ious_indicator = (ious == ious[max_ind]).astype(np.float)
                max_ious_indices = np.nonzero(max_ious_indicator)[0]
                max_ind = max_ious_indices[int((len(max_ious_indices) - 1)/2)]

                yolo_layer_ind, anchor_ind, row, col = self.bbox_info[max_ind]
                objectness_mask[yolo_layer_ind][i][anchor_ind][row][col] = 2



                # set objectness target 
                objectness_target[yolo_layer_ind][i][anchor_ind][row][col] = 1

                # set coordinates target
                gt_box = label[label_ind][1:]
                center_x = (gt_box[0] + 0.5 * gt_box[2]) * self.yolo_layer_size[yolo_layer_ind]
                center_y = (gt_box[1] + 0.5 * gt_box[3]) * self.yolo_layer_size[yolo_layer_ind]
                bw = gt_box[2] * float(self.yolov3.net_info['width'])
                bh = gt_box[3] * float(self.yolov3.net_info['height'])
                pw = self.anchor_sizes[yolo_layer_ind][anchor_ind][0]
                ph = self.anchor_sizes[yolo_layer_ind][anchor_ind][1]
                delta_x = center_x - col
                delta_y = center_y - row

                ## [todo]: review this part
                delta_x = max(0.000001, delta_x)
                delta_y = max(0.000001, delta_y)
                delta_x = min(0.999999, delta_x)
                delta_y = min(0.999999, delta_y)

                coordinate_target[yolo_layer_ind][i][anchor_ind * 4][row][col] = math.log(delta_x / (1.0 - delta_x))
                coordinate_target[yolo_layer_ind][i][anchor_ind * 4 + 1][row][col] = math.log(delta_y / (1.0 - delta_y))
                try:
                    coordinate_target[yolo_layer_ind][i][anchor_ind * 4 + 2][row][col] = math.log(bw/pw)
                    coordinate_target[yolo_layer_ind][i][anchor_ind * 4 + 3][row][col] = math.log(bh/ph)
                except:
                    print()
                    print('!!!!warning!!!, bw %f pw %f bh %f ph %f' % (bw, pw, bh, ph))
                    continue

                # set coordinates mask
                coordinate_mask[yolo_layer_ind][i, anchor_ind * 4 : (anchor_ind + 1) * 4, row, col] = 1

                # set class target
                category = int(label[label_ind][0])
                class_target[yolo_layer_ind][i][anchor_ind * self.classes + category][row][col] = 1.0
                
                # set class mask
                class_mask[yolo_layer_ind][i, anchor_ind * self.classes : (anchor_ind + 1) * self.classes, row, col] = 1


        for i in range(0, yolo_layers):
            objectness_mask[i] = objectness_mask[i] > 0
            objectness_mask[i] = objectness_mask[i].to(torch.float32)

            if self.use_cuda:
                objectness_target[i] = objectness_target[i].cuda()
                objectness_mask[i] = objectness_mask[i].cuda()
                coordinate_target[i] = coordinate_target[i].cuda()
                coordinate_mask[i] = coordinate_mask[i].cuda()
                class_target[i] = class_target[i].cuda()
                class_mask[i] = class_mask[i].cuda()

        objectness_loss = 0.0
        coordinate_loss = 0.0
        class_loss = 0.0

        for i in range(0, yolo_layers):
            # objectness loss
            objectness_ouptut = torch.sigmoid(yolo_output[i][:, 4::(self.classes+5), :, :])
            objectness_loss += self.bce_loss(objectness_ouptut * objectness_mask[i], 
                    objectness_target[i]*objectness_mask[i])
            
            # coordinate loss
            coordinate_ind_template = [0, 1, 2, 3]
            coordinate_dimension_ind = []
            for anchor_ind in range(0, self.anchors[i]):
                coordinate_dimension_ind += [ind + (self.classes+5)*anchor_ind for ind in coordinate_ind_template]
            coordinate_output = yolo_output[i][:, coordinate_dimension_ind, :, :]
            coordinate_loss += self.mse_loss(coordinate_output * coordinate_mask[i], coordinate_target[i] * coordinate_mask[i])
            # category loss
            class_ind_template = range(5, 5 + self.classes)
            class_dimension_ind = []
            for anchor_ind in range(0, self.anchors[i]):
                class_dimension_ind += [ind + (self.classes + 5)*anchor_ind for ind in class_ind_template]
            class_output = torch.sigmoid(yolo_output[i][:, class_dimension_ind, :, :])
            class_loss += self.bce_loss(class_output * class_mask[i], class_target[i] * class_mask[i])

        return objectness_loss, coordinate_loss, class_loss


    def compute_max_iou_bbox_index(self, labels):
        '''
        labels are in shape (1, 50, 5), the 5 columns are (label, x, y, w, h)
        if lable < 0, then it's a fillup, not real groud truth label
        '''

        pass


    def get_bbox_priors(self):
        input_img_size = int(self.yolov3.module_dict_list[0]['width'])
        bbox_priors = []
        self.bbox_info = [] # store yolo layer index, row, col
        yolo_layer_index = 0
        for module in self.yolov3.module_dict_list:
            if module['type'] != 'yolo':
                continue
            self.classes = int(module['classes'])
            yolo_layer_size = int(input_img_size / int(module['scaling_ratio']))
            self.yolo_layer_size.append(yolo_layer_size)
            mask = module['mask']
            mask = mask.strip(' ').split(',')
            mask = [int(i) for i in mask]
            anchors = module['anchors'].strip(' ').split(', ')
            anchors = [anchors[i] for i in mask]
            anchors = [[int(j) for j in i.strip(' ').split(',')] for i in anchors]
            self.anchor_sizes.append(anchors)
            self.anchors.append(len(mask))
            for i in range(0, yolo_layer_size):
                for j in range(0, yolo_layer_size):

                    rect_center_x = (float(j) + 0.5) / float(yolo_layer_size)
                    rect_center_y = (float(i) + 0.5) / float(yolo_layer_size)
                    for anchor_ind, anchor in enumerate(anchors):
                        rect_w = float(anchor[0]) / float(input_img_size)
                        rect_h = float(anchor[1]) / float(input_img_size)

                        bbox_priors.append((rect_center_x - rect_w * 0.5,
                            rect_center_y - rect_h* 0.5,
                            rect_w, rect_h))
                        self.bbox_info.append((yolo_layer_index, anchor_ind, i, j))
            self.bbox_priors = np.asarray(bbox_priors)
            yolo_layer_index += 1

if __name__ == '__main__':
    
    # param
    epochs=30
    print_every = 10
    test_every = 200
    batch_size = 12
    ## loss weight
    objectness_loss_weight = 1.0
    coordinate_loss_weight = 1.0
    class_loss_weight = 30.0

    learning_rate = 0.0005
    obj_threshold = 0.3

    
    trainer = YoloV3Trainer('./yolov3.cfg', './data/darknet53.conv.imagenet.74')

    coco_train_dataset = CocoDataset('/home/bigning/hdd/dataset/coco2017',
            './data/', 'train', 416)

    train_dataloader = torch.utils.data.DataLoader(coco_train_dataset,
            batch_size = batch_size, shuffle=True, num_workers=4)
            #batch_size = 1, shuffle=True, num_workers=2)

    val_img_path = '/home/bigning/hdd/dataset/coco2017/val2017/'
    val_img_names = os.listdir(val_img_path)
    random.shuffle(val_img_names)
    val_img_ind = 0

    ## load trained model
    check_point = torch.load('./model_epoch_20')
    trainer.yolov3.load_state_dict(check_point['model'])

    use_cuda = torch.cuda.is_available()
    print('use cuda: {}'.format(use_cuda))
    if use_cuda:
        trainer.cuda()
    trainer.train()

    optimizer = torch.optim.SGD(trainer.parameters(), lr=learning_rate, momentum=0.9)
    for i in range(20, epochs):
        running_loss = 0.0
        start = time.time()
        for ind, (resized_imgs, labels) in enumerate(train_dataloader):
            resized_imgs.detach()
            if use_cuda:
                resized_imgs = resized_imgs.cuda()

            optimizer.zero_grad()
            yolo_output = trainer.forward(resized_imgs)
            objectness_loss, coordinate_loss, class_loss = trainer.compute_loss(yolo_output, labels)
            loss = objectness_loss_weight * objectness_loss + coordinate_loss_weight * coordinate_loss + \
                class_loss_weight * class_loss

            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if ind % print_every == print_every - 1:
                end_time = time.time() - start
                avg_time = end_time / print_every
                avg_loss = running_loss / print_every
                print('epoch [%d], batch [%d], avg_loss: %f, time: [%f]' % (i, ind, avg_loss, avg_time))
                print(' objectness loss: %f, coord loss: %f, class loss: %f' % (objectness_loss, coordinate_loss, class_loss))
                print()

                running_loss = 0.0
                start = time.time()
            if ind % test_every == test_every - 1:
                trainer.eval()
                val_img_fullname = val_img_path + val_img_names[val_img_ind%len(val_img_names)]
                val_img = cv2.imread(val_img_fullname)
                val_img_ind += 1
                result_img = detect_image(val_img, trainer.yolov3, use_cuda, show_img=False, nms=False, obj_threshold=obj_threshold)
                new_name = './data/eval_result/%d_%d.jpg' % (i, ind) 
                cv2.imwrite(new_name, result_img)
                trainer.train()

        save_dict = {}
        save_dict['model'] = trainer.yolov3.state_dict()
        torch.save(save_dict, 'model_epoch_%d' % i)
