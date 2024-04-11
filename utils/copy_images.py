import os
import shutil
from pathlib import Path
from tqdm import tqdm

cwd = os.getcwd()
OLD_IMAGE_FOLDER = 'laion_face_data/glasses_imgs'
NEW_IMAGE_FOLDER = 'laion_face_data/selected_glasses_imgs' # files with adjectives in text
os.makedirs(
    name=f'{cwd}/{NEW_IMAGE_FOLDER}',
    exist_ok=True
)

for file_name in tqdm(os.listdir(f'{cwd}/{OLD_IMAGE_FOLDER}')):
    image_name = file_name.split(".")[0] + '.png'
    
    # makes sure we dont have duplicates
    if (Path(f"{OLD_IMAGE_FOLDER}/{image_name}").is_file()) and not(Path(f"{NEW_IMAGE_FOLDER}/{image_name}").is_file()):
        shutil.copyfile(f"{OLD_IMAGE_FOLDER}/{image_name}", f"{NEW_IMAGE_FOLDER}/{image_name}")