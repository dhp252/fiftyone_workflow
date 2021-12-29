import os
from glob import glob
import argparse

from tqdm import tqdm
import albumentations as aug
import cv2
import matplotlib.pyplot as plt

# Augment config
transform = aug.Compose([
    aug.RandomBrightnessContrast(brightness_limit=(-0.4, 0.2), p=1),
    aug.HueSaturationValue(hue_shift_limit=16, \
                        sat_shift_limit=16, \
                        val_shift_limit=16, \
                        p=1),
    aug.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
    # aug.RGBShift(r_shift_limit=16, g_shift_limit=16, b_shift_limit=16, p=1),
    aug.ToGray(p=0.2),
    # aug.RandomGamma(p=1),
    # aug.ChannelShuffle(p=1),
    aug.MotionBlur(p=0.5, blur_limit=9),
    aug.GaussNoise(var_limit=(30.0, 100.0),p=0.7),
    # aug.CoarseDropout(max_height=2, max_width=2, max_holes=16, min_holes=4, fill_value=[0,0,0], p=1),
    aug.Perspective(scale=(0.01, 0.1), keep_size=False, fit_output=True, p=0.7),
    aug.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.7),
    aug.HorizontalFlip(p=0.5),

    # aug.augmentations.geometric.transforms.Affine (rotate=(-180, 180), fit_output=True, p=0.5)
], bbox_params=aug.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))


def augment(src_img_folder, src_label_folder, dst_img_folder, dst_label_folder, n=2):

    os.makedirs(dst_img_folder, exist_ok=True)
    os.makedirs(dst_label_folder, exist_ok=True)

    image_paths = sorted(glob(f'{src_img_folder}/*.jpg'))
    
    for path in tqdm(image_paths):
        # read original image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        name = path.split('/')[-1].split('.')[0]

        # read original label
        f = open(f'{src_label_folder}/{name}.txt')
        content = f.read().splitlines()
        f.close()
        bboxes = []
        class_labels = []
        for label in content:
            label = label.split(' ')
            class_id = int(label[0])
            label = list(map(float, label[1:]))
            bboxes.append(label)
            class_labels.append(class_id)
        # breakpoint()
        for i in range(n):
            # augment i-th time
            try:
                transformed_image = transform(image=image,
                                              bboxes=bboxes, 
                                              class_labels=class_labels)
            except:
                continue

            # save new image
            cv2.imwrite(f'{dst_img_folder}/{name}_aug_{i}.jpg', transformed_image['image'])

            # save new label
            label_txt = ""
            for j, class_id in enumerate(transformed_image['class_labels']):
                bbox = transformed_image['bboxes'][j]
                bbox = ' '.join([str(b) for b in bbox])
                label_txt += str(class_id) + ' ' + bbox + '\n'

            f = open(f'{dst_label_folder}/{name}_aug_{i}.txt', 'w')
            f.write(label_txt)
            f.close()

        
if __name__=="__main__":
    
    
    
    N_AUGMENT_PER_IMG = 2
    # SRC_IMG_FOLDER    = 'dataset_22_11_2021/train/images'
    # SRC_LABEL_FOLDER  = 'dataset_22_11_2021/train/labels'
    # DST_IMG_FOLDER    = 'dataset_22_11_2021/train_aug/images'
    # DST_LABEL_FOLDER  = 'dataset_22_11_2021/train_aug/labels'
    SRC_IMG_FOLDER    = 'test_export/images/validation'
    SRC_LABEL_FOLDER  = 'test_export/labels/validation'
    DST_IMG_FOLDER    = 'test_export/images/validation'
    DST_LABEL_FOLDER  = 'test_export/labels/validation'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-si','--src_img',   default=SRC_IMG_FOLDER)
    parser.add_argument('-sl','--src_label', default=SRC_LABEL_FOLDER)
    parser.add_argument('-di','--dst_img',   default=DST_IMG_FOLDER)
    parser.add_argument('-dl','--dst_label', default=DST_LABEL_FOLDER)
    parser.add_argument('-n','--number',     default=DST_LABEL_FOLDER, type=int)
    
    args = parser.parse_args()
    
    print('Start Augment', args.src_img, 
          args.src_label, args.dst_img, 
          args.dst_label, args.number)
    
    augment(
        src_img_folder   = args.src_img,
        src_label_folder = args.src_label,
        dst_img_folder   = args.dst_img,
        dst_label_folder = args.dst_label,
        n                = args.number,
    )