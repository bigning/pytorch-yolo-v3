import os,sys
import torch.nn
import torch
import darknet
import cv2
import numpy as np
import time
import math
import copy

def detect_image(img, yolo_v3, use_cuda, show_img=True, nms=True, obj_threshold=0.5):
    class_name_file = open('coco.names')
    names = class_name_file.readlines()
    class_name_file.close()
    names = [name.strip('\n') for name in names]

    img_size = int(yolo_v3.net_info['width'])
    img_var = cv2.resize(img, (img_size, img_size))
    img_var = img_var[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_var = img_var[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_var = torch.from_numpy(img_var).float()
    if use_cuda:
        img_var = img_var.cuda()
    start = time.time()

    with torch.no_grad():
        detection = yolo_v3(img_var)
    #detection = yolo_v3(img_var)
    end = time.time()
    print('time: {}'.format(end - start))
    detection = detection.detach().cpu().numpy()

    # draw result
    stride_x = float(img.shape[1]) / float(img_size)
    stride_y = float(img.shape[0]) / float(img_size)
    objectness_threshold = obj_threshold
    detection = detection[0,:,:]
    bboxes = {}
    scores = {}
    all_indices = {}
    class_indices = {}
    nms_threshold = 0.5
    for i in range(detection.shape[0]):
        objectness = detection[i, 4]
        if objectness > objectness_threshold:
            center_x = detection[i, 0] * stride_x
            center_y = detection[i, 1] * stride_y
            w = detection[i, 2] * stride_x
            h = detection[i, 3] * stride_y
            x = max(0, int(math.floor(center_x - 0.5 * w)))
            y = max(0, int(math.floor(center_y - 0.5 * h)))
            w = int(math.floor(w))
            h = int(math.floor(h))

            classes = detection[i, 5:]
            class_index = np.argmax(classes)
            class_name = names[class_index]

            if class_name in bboxes:
                bboxes[class_name].append([x, y, w, h])
                scores[class_name].append(float(objectness))
                all_indices[class_name].append(i)
                class_indices[class_name].append(class_index)
            else:
                bboxes[class_name] = [[x, y, w, h]]
                scores[class_name] = [float(objectness)]
                all_indices[class_name] = [i]
                class_indices[class_name] = [class_index]
    for class_name in bboxes:
        if nms:
            indices = cv2.dnn.NMSBoxes(bboxes[class_name], scores[class_name], objectness_threshold, nms_threshold)
        else:
            indices = range(len(bboxes[class_name]))
        for ii in indices:
            i = int(ii)
            x = bboxes[class_name][i][0]
            y = bboxes[class_name][i][1]
            w = bboxes[class_name][i][2]
            h = bboxes[class_name][i][3]
           
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255)) 

            # get class and class prob
            index = all_indices[class_name][i]
            class_index = class_indices[class_name][i]
            prob = detection[index, 5:][class_index]
            #print(detection[index, 5:])
            text = class_name + ': {}'.format(prob)
            cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    if show_img:
        cv2.imshow("a", img)
        cv2.waitKey()
    return img

if __name__ == '__main__':
    yolo_v3 = darknet.DarknetYoloV3('./yolov3.cfg')
    
    check_point = torch.load('./model_epoch_20')
    yolo_v3.load_state_dict(check_point['model'])
    
    #yolo_v3.load_weight('./data/yolov3.weights')
    #yolo_v3.load_weight('./data/darknet53.conv.imagenet.74', True)
    use_cuda = torch.cuda.is_available()
    print('use cuda: {}'.format(use_cuda))
    
    if use_cuda:
        yolo_v3.cuda()
    yolo_v3.eval()
    print('model is ready')
    #img = cv2.imread('./data/images/7f17d07dcc75de3a.jpg')
    img = cv2.imread('./data/images/7f1c4dee402c50de.jpg')

    detect_image(img, yolo_v3, use_cuda, obj_threshold=0.3)
