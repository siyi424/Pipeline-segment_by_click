import os
import numpy as np
from PIL import Image
import pydicom
from tqdm import tqdm
from config import dicom_dir, mask_dir, output_dir

def load_dicom(dicom_path):
    return pydicom.dcmread(dicom_path)

def load_mask(mask_path):
    mask_image = Image.open(mask_path).convert('L')  # Convert to grayscale
    return np.array(mask_image)

def apply_mask_to_dicom(dicom, mask):
    original_pixel_array = dicom.pixel_array
    masked_pixel_array = np.where(mask == 255, original_pixel_array, 0)
    dicom.PixelData = masked_pixel_array.astype(np.uint16).tobytes()
    return dicom

def save_dicom(dicom, output_path):
    dicom.save_as(output_path)

def main(dicom_dir, mask_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dicom_files = sorted(os.listdir(dicom_dir))
    for dicom_filename in tqdm(dicom_files, desc="Processing DICOM files"):
        dicom_path = os.path.join(dicom_dir, dicom_filename)
        mask_filename = dicom_filename.replace('.dcm', '.jpg')
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if os.path.exists(mask_path):
            dicom = load_dicom(dicom_path)
            mask = load_mask(mask_path)
            dicom = apply_mask_to_dicom(dicom, mask)
            output_path = os.path.join(output_dir, dicom_filename)
            save_dicom(dicom, output_path)
        else:
            print(f"Mask for {dicom_filename} not found at {mask_path}")

# 请确保在调用 main 函数时传入正确的 dicom_dir, mask_dir 和 output_dir 参数
main(dicom_dir, mask_dir, output_dir)