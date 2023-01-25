import os
from tqdm import tqdm

# EXE_PATH = '/Users/mitya/Library/Developer/Xcode/DerivedData/FaceLandmarks-bhrustcsbywvfydenfbdrkuouial/Build/Products/Debug/FaceLandmarks'
EXE_PATH = 'echo'
OUTPUT_DIR = './annotations'

def annotate_image(image_path, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    code = os.system(f'{EXE_PATH} "{image_path}" > "{json_path}"')
    if code != 0:
        print(f'Error in {image_path}')
        exit(1)

def annotate_300W_WFLW():
    BASE_FOLDER = '/Users/mitya/datasets'
    folders = [
        '300W/afw',
        '300W/helen/trainset', '300W/helen/testset',
        '300W/lfpw/trainset', '300W/lfpw/testset',
        '300W/ibug',
        *['WFLW_images/'+d for d in os.listdir(os.path.join(BASE_FOLDER, 'WFLW_images'))],
    ]

    for folder in tqdm(folders):
        dataset_name = folder.split('/')[0]
        output_folder = os.path.join(OUTPUT_DIR, dataset_name, *folder.split('/')[1:])
        for file in tqdm(os.listdir(os.path.join(BASE_FOLDER, folder)), desc=folder, leave=False):
            name, ext = os.path.splitext(file)
            if ext not in ['.jpg', '.jpeg', '.png']:
                continue
            full_image_path = os.path.join(BASE_FOLDER, folder, file)
            full_output_path = os.path.join(output_folder, name+'.json')
            annotate_image(full_image_path, full_output_path)

def annotate_ffhq():
    DATASET_FOLDER = '/Volumes/EXTERNALHDD/ffhq-dataset/images1024x1024'

    for image_inx in tqdm(range(70_000)):
        folder_name = f'{(image_inx // 1_000):02d}000'
        image_name = f'{image_inx:05d}'
        full_image_path = os.path.join(DATASET_FOLDER, folder_name, image_name + '.png')
        full_output_path = os.path.join(OUTPUT_DIR, 'ffhq', folder_name, image_name + '.json')
        annotate_image(full_image_path, full_output_path)

def annotate_celeba_hq():
    DATASET_FOLDER = '/Users/mitya/datasets/celeba_hq'

    for split in ['train', 'val']:
        for gender in ['male', 'female']:
            files = os.listdir(os.path.join(DATASET_FOLDER, split, gender))
            for image_name in tqdm(files, leave=False, desc=f'{split} {gender}'):
                extension = image_name.split('.')[-1]
                if extension not in ['jpg', 'jpeg', 'png']:
                    continue

                json_name = image_name[:-len(extension)] + 'json'
                full_image_path = os.path.join(DATASET_FOLDER, split, gender, image_name)
                full_output_path = os.path.join(OUTPUT_DIR, 'celeba_hq', split, gender, json_name)
                annotate_image(full_image_path, full_output_path)
