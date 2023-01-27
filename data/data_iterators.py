from dataclasses import dataclass
from typing import List, Tuple
import os
import json
from itertools import chain
from PIL import Image

@dataclass
class Annotation:
    """
    Bounding box is in format [left, top, right, bottom]
    Landmarks are in format [(x1, y1), (x2, y2), ...]
    all coordinates are in pixels (image dimensions)
    """
    bbox: List[int] # [left, top, right, bottom]
    landmarks: List[Tuple[int, int]] # [ (x1, y1), (x2, y2), ... ]

@dataclass
class ImageWithFaces:
    image_path: str
    image_size: Tuple[int, int]
    annotation_path: str
    faces: List[Annotation]

def __make_pair(image_path, annotation_path):
    assert os.path.isfile(image_path), f'Image {image_path} does not exist'
    assert os.path.isfile(annotation_path), f'Annotation {annotation_path} does not exist'
    with open(annotation_path) as f:
        raw_annotations = json.load(f)
    width, height = Image.open(image_path).size
    annotations = []
    for face in raw_annotations:
        left, top, right, bottom = [face['bbox'][k] for k in ['left', 'top', 'right', 'bottom']]
        left, top, right, bottom = int(left * width), int(top * height), int(right * width), int(bottom * height)
        bbox = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
        box_width = right - left
        box_height = bottom - top
        landmarks = [(int(l[0] * box_width + left), int(l[1] * box_height + top)) for l in face['landmarks']]
        annotations.append(Annotation(bbox, landmarks))
    return ImageWithFaces(image_path, (width, height), annotation_path, annotations)

def __simple(images_path, annotations_path, sub_folder, img_format='.jpg') -> List[ImageWithFaces]:
    images_path = os.path.join(images_path, sub_folder)
    annotations_path = os.path.join(annotations_path, sub_folder)

    annotation_files = os.listdir(annotations_path)
    assert all([f.endswith('.json') for f in annotation_files])
    for file in annotation_files:
        image_path = os.path.join(images_path, file[:-5] + img_format)
        annotation_path = os.path.join(annotations_path, file)
        yield __make_pair(image_path, annotation_path)

def get_300W_afw(images_path, annotations_path):
    return __simple(images_path, annotations_path, '300W/afw')

def get_300W_ibug(images_path, annotations_path):
    return __simple(images_path, annotations_path, '300W/ibug')

def get_300W_helen_train(images_path, annotations_path):
    return __simple(images_path, annotations_path, '300W/helen/trainset')
def get_300W_helen_test(images_path, annotations_path):
    return __simple(images_path, annotations_path, '300W/helen/testset')
def get_300W_helen_all(images_path, annotations_path):
    return chain(get_300W_helen_train(images_path, annotations_path),
        get_300W_helen_test(images_path, annotations_path))

def get_300W_lfpw_train(images_path, annotations_path):
    return __simple(images_path, annotations_path, '300W/lfpw/trainset', '.png')
def get_300W_lfpw_test(images_path, annotations_path):
    return __simple(images_path, annotations_path, '300W/lfpw/testset', '.png')
def get_300W_lfpw_all(images_path, annotations_path):
    return chain(get_300W_lfpw_train(images_path, annotations_path),
        get_300W_lfpw_test(images_path, annotations_path))

def get_300W_all(images_path, annotations_path):
    return chain(get_300W_afw(images_path, annotations_path),
           get_300W_ibug(images_path, annotations_path),
           get_300W_helen_all(images_path, annotations_path),
           get_300W_lfpw_all(images_path, annotations_path))

def get_celeba_train(images_path, annotations_path):
    return chain(__simple(images_path, annotations_path, 'celeba_hq/train/female'),
        __simple(images_path, annotations_path, 'celeba_hq/train/male'))
def get_celeba_val(images_path, annotations_path):
    return chain(__simple(images_path, annotations_path, 'celeba_hq/val/female'),
        __simple(images_path, annotations_path, 'celeba_hq/val/male'))
def get_celeba_all(images_path, annotations_path):
    return chain(get_celeba_train(images_path, annotations_path),
        get_celeba_val(images_path, annotations_path))

def get_ffhq(images_path, annotations_path):
    images_path = os.path.join(images_path, 'ffhq-dataset/images1024x1024')
    annotations_path = os.path.join(annotations_path, 'ffhq')
    for idx in range(70000):
        folder_name = f'{(idx // 1_000):02d}000'
        item_name = f'{idx:05d}'
        image_path = os.path.join(images_path, folder_name, item_name + '.png')
        annotation_path = os.path.join(annotations_path, folder_name, item_name + '.json')
        yield __make_pair(image_path, annotation_path)

def get_WFLW(images_path, annotations_path):
    images_path = os.path.join(images_path, 'WFLW/WFLW_images')
    annotations_path = os.path.join(annotations_path, 'WFLW')
    subfolders = os.listdir(annotations_path)
    for subfolder in subfolders:
        files = os.listdir(os.path.join(annotations_path, subfolder))
        for file in files:
            image_path = os.path.join(images_path, subfolder, file[:-5] + '.jpg')
            annotation_path = os.path.join(annotations_path, subfolder, file)
            yield __make_pair(image_path, annotation_path)

def get_all(images_path, annotations_path):
    return chain(get_300W_all(images_path, annotations_path),
        get_celeba_all(images_path, annotations_path),
        get_ffhq(images_path, annotations_path),
        get_WFLW(images_path, annotations_path))

if __name__ == '__main__':
    IMAGES_DIR = '/media/pupa/DataStorage/datasets'
    ANNOTATIONS_DIR = '/home/pupa/dev/apple_face/annotations'
    image_count = 0
    face_count = 0
    for data in get_all(IMAGES_DIR, ANNOTATIONS_DIR):
        image_count += 1
        face_count += len(data.faces)
    print(f'Images: {image_count}, faces: {face_count}')
