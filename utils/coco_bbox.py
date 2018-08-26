import os,sys, random
sys.path.append('/home/bigning/hdd/projects/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import cv2
import pickle

def extract_and_save_bbox(ann_file, train_or_val):
    coco=COCO(ann_file)
    cat_names = []
    cat_id_to_compact_id = {}
    cats = coco.loadCats(coco.getCatIds())
    for index, cat in enumerate(cats):
        cat_names.append(cat['name'])
        cat_id_to_compact_id[cat['id']] = index
    f = open('../data/coco_names_{}'.format(train_or_val), 'w')
    for name in cat_names:
        f.writelines(name + '\n')
    f.close()

    img_bbox = {}
    imgIds = coco.getImgIds()
    annIds = coco.getAnnIds(imgIds=imgIds)
    anns = coco.loadAnns(annIds)
    for ann in anns:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        label = cat_id_to_compact_id[cat_id]
        bbox = ann['bbox']
        gt = (label, bbox)
        if img_id not in img_bbox:
            img_bbox[img_id] = [gt]
        else:
            img_bbox[img_id].append(gt)
    f = open('../data/coco_{}.label'.format(train_or_val), 'wb')
    pickle.dump(img_bbox, f)
    f.close()

def verify_random_examples():
    names_file = open('../data/coco_names_train')
    names = names_file.readlines()
    names_file.close()
    names = [name.strip('\n') for name in names]
    train_img_path = '/home/bigning/hdd/dataset/coco2017/train2017/'
    filenames = os.listdir(train_img_path)
    random.shuffle(filenames)
    f = open('../data/coco_train.label', 'rb')
    gt_dict = pickle.load(f)
    f.close()
    no_obj_imgs = 0
    for filename in filenames[:100]:
        img = cv2.imread(train_img_path + filename)
        filename_arr = filename.split('.')
        img_id = int(filename_arr[0])
        if img_id not in gt_dict:
            no_obj_imgs += 1
            continue
        gts = gt_dict[img_id]
        for gt in gts:
            label = names[gt[0]]
            bbox = gt[1]
            bbox = [int(f) for f in bbox]
            x = bbox[0]
            y = bbox[1]
            cv2.putText(img, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x,y), (x + bbox[2], y + bbox[3]), (0, 0, 255))
        cv2.imwrite('../data/coco_examples/{}'.format(filename), img)
    print(no_obj_imgs)

if __name__=='__main__':
    train_ann_file = '/home/bigning/hdd/dataset/coco2017/annotations/instances_train2017.json'
    val_ann_file = '/home/bigning/hdd/dataset/coco2017/annotations/instances_val2017.json'
    #extract_and_save_bbox(val_ann_file, 'val')
    #extract_and_save_bbox(train_ann_file, 'train')

    verify_random_examples()
