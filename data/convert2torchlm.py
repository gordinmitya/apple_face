from typing import Optional, List
import os
import cv2
import numpy as np
from tqdm import tqdm

import data_iterators as di

# TODO get rid of hacky imports
import sys
cwf = os.path.dirname(__file__)
torchlm_path = '/'.join([*cwf.split('/')[:-1], 'torchlm'])
sys.path.append(torchlm_path)
from torchlm import transforms
from torchlm.data._annotools import format_annotation

IMAGES_DIR = '/media/pupa/DataStorage/datasets'
ANNOTATIONS_DIR = '/home/pupa/dev/apple_face/annotations'

def process_set(
    split_name: str,
    split_data: List[di.ImageWithFaces], 
    save_dir: str,
    extend: Optional[float] = 0.2, # padding
    target_size: int = 256,
    keep_aspect: Optional[bool] = False,
    force_normalize: Optional[bool] = True,
):
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    landmarks_path = os.path.join(save_dir, f'{split_name}.txt')
    landmarks_file = open(landmarks_path, 'w')
    resize_op = transforms.LandmarksResize((target_size, target_size), keep_aspect)
    for item in tqdm(split_data):
        img = cv2.imread(item.image_path)
        img_width, img_height = img.shape[1], img.shape[0]
        for face_idx, face in enumerate(item.faces):
            landmarks = np.array(face.landmarks, dtype=np.float32)
            xmin, xmax = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
            ymin, ymax = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

            # padding
            crop_width = xmax - xmin
            crop_height = ymax - ymin
            xmin -= int(crop_width * extend / 2.)
            ymin -= int(crop_height * extend / 2.)
            xmax += int(crop_width * extend / 2.)
            ymax += int(crop_height * extend / 2.)
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(img_width - 1, xmax), min(img_height - 1, ymax)
            crop_width = xmax - xmin
            crop_height = ymax - ymin

            crop = img[int(ymin):int(ymax), int(xmin):int(xmax), :].copy()
            # adjust according to left-top corner
            landmarks[:, 0] -= float(xmin)
            landmarks[:, 1] -= float(ymin)
            crop, landmarks = resize_op(crop, landmarks)
            crop_height, crop_width, _ = crop.shape

            if force_normalize:
                landmarks[:, 0] /= float(crop_width)
                landmarks[:, 1] /= float(crop_height)
            
            suffix = '.png'
            if len(item.faces) > 1:
                suffix = f'_{face_idx}.jpg'
            sample_name = item.image_path[len(IMAGES_DIR.rstrip('/')) + 1:]
            sample_name = os.path.splitext(sample_name)[0] + suffix
            crop_path = os.path.join(save_dir, sample_name)

            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            cv2.imwrite(crop_path, crop)
            landmarks_file.write(format_annotation(sample_name, landmarks) + '\n')

    landmarks_file.close()

if __name__ == '__main__':
    train_data = list(di.get_train(IMAGES_DIR, ANNOTATIONS_DIR))
    process_set('train', train_data, os.path.join(IMAGES_DIR, 'apple_face_crops/train'))

    test_data = list(di.get_test(IMAGES_DIR, ANNOTATIONS_DIR))
    process_set('test', test_data, os.path.join(IMAGES_DIR, 'apple_face_crops/test'))
