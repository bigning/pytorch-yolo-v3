import os,sys
import torch.nn
import torch
import darknet
import cv2
import numpy as np
import time

def detect_image(img, yolo_v3):
    img_size = int(yolo_v3.net_info['width'])
    img_var = cv2.resize(img, (img_size, img_size))
    img_var = img_var[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_var = img_var[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_var = torch.from_numpy(img_var).float().detach()
    start = time.time()
    with torch.no_grad():
        detection = yolo_v3(img_var)
    end = time.time()
    print('time: {}'.format(end - start))
    detection = detection.detach().numpy()
    print(detection.shape)
    

if __name__ == '__main__':
    yolo_v3 = darknet.DarknetYoloV3('./yolov3.cfg')
    yolo_v3.load_weight('./data/yolov3.weights')
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        yolo_v3.cuda()
    
    yolo_v3.eval()
    print('model is ready')
    img = cv2.imread('./data/images/7f17d07dcc75de3a.jpg')
    

    detect_image(img, yolo_v3)
    cv2.imshow("a", img)
    cv2.waitKey()
