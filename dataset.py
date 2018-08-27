import torch, cv2
import os
import pickle
from torch.utils.data import Dataset

def img_preprocess(img, img_size):
    resize_img = cv2.resize(img, (self.img_size, self.img_size))
    # BGR -> RGB | H W C -> C H W
    resize_img = resize_img[:, :, ::-1].transpose((2, 0, 1))
    resize_img = resize_img / 255.0
    resize_img = torch.from_numpy(resize_img).float()
    return resize_img


class CocoDataset(torch.utils.data.Dataset, img_size):
    def __init__(self, img_root_path, label_root_path, train_or_eval):
        self.img_root_path = img_root_path + '/{}2017/'.format(train_or_eval)
        self.img_names = os.listdir(self.img_root_path)

        label_file_name = label_root_path + '/coco_{}.label'.format(train_or_eval)
        label_file = open(label_file_name, 'rb')
        self.gts_dict = pickle.load(label_file)
        label_file.close()

        self.img_size = img_size
        self.train_or_eval = train_or_eval

        super(CocoDataset, self).__init__()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # img
        img_name = self.img_names[index]
        original_img = cv2.imread(self.img_root_path + img_name)
        original_img_width = float(original_img.shape[1])
        original_img_height = float(original_img.shape[0])
        resized_img_tensor = img_preprocess(original_img, self.img_size)

        # labels [(class_id, [x,y,w,h])]
        # NOTE!!! here, x and y is the upper-left coordinates of the bbox,
        # it's not the center point!!!!
        labels = []
        img_name_arr = img_name.split('.')
        img_id = int(img_name_arr[0])
        
        if img_id in self.gts_dict:
            for gt in self.gts_dict[img_id]:
                class_id = gt[0]
                bbox = gt[1]
                # normailize bbox
                normalized_bbox = [bbox[0]/original_img_width,
                                   bbox[1]/original_img_height,
                                   bbox[2]/original_img_width,
                                   bbox[3]/original_img_height]
                labels.append((class_id, normalized_bbox))
        return resized_img_tensor, labels
