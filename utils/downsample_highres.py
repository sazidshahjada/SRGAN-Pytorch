import cv2
import os
import glob
from tqdm import tqdm

input_folder = "/home/iot/SRGAN_implementation/samples/high_res_image"
output_folder = "/home/iot/SRGAN_implementation/samples/low_res_image"
downscale_factor = 4

os.makedirs(output_folder, exist_ok=True)
img_paths = glob.glob(os.path.join(input_folder, "*.*"))
for img_path in tqdm(img_paths):
    img = cv2.imread(img_path)
    lr_img = cv2.resize(img, (img.shape[1] // downscale_factor, img.shape[0] // downscale_factor), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), lr_img)
