import os
import pydicom
import numpy as np
import cv2
from tqdm import tqdm
from config import input_post_dcms, input_pre_dcms, HU_post_jpgs, HU_pre_jpgs

def hu_to_binary_mask(dcm_file, hu_threshold):
    # 读取DCM文件
    ds = pydicom.dcmread(dcm_file)
    img = ds.pixel_array
    # 将像素值转换为HU值
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    hu_img = img * slope + intercept
    # 生成二值mask
    binary_mask = (hu_img >= hu_threshold).astype(np.uint8)
    return binary_mask

def save_mask_as_jpg(mask, output_path):
    # 将二值mask保存为JPG格式
    cv2.imwrite(output_path, mask * 255)

def process_dcm_files(input_folder, output_folder, hu_threshold):
    # 获取所有DCM文件
    dcm_files = [os.path.join(root, file) for root, _, files in os.walk(input_folder) for file in files if file.endswith('.dcm')]
    # 遍历DCM文件并显示进度条
    for dcm_file in tqdm(dcm_files, desc="Processing DCM files"):
        # 获取序列号
        ds = pydicom.dcmread(dcm_file)
        series_number = ds.SeriesNumber
        # 生成二值mask
        binary_mask = hu_to_binary_mask(dcm_file, hu_threshold)
        # 构建输出路径，保持文件名不变，只是格式改为JPG
        output_path = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(dcm_file))[0]}.jpg')
        # 保存二值mask为JPG格式
        save_mask_as_jpg(binary_mask, output_path)

if __name__ == "__main__":
    # post_contrast
    input_folder = input_post_dcms  # 输入文件夹路径
    output_folder = HU_post_jpgs  # 输出文件夹路径
    hu_threshold = 1676  # 自定义HU值阈值
    os.makedirs(output_folder, exist_ok=True)
    process_dcm_files(input_folder, output_folder, hu_threshold)

    # pre_contrast
    input_folder = input_pre_dcms  # 输入文件夹路径
    output_folder = HU_pre_jpgs  # 输出文件夹路径
    hu_threshold = 1024  # 自定义HU值阈值
    os.makedirs(output_folder, exist_ok=True)
    process_dcm_files(input_folder, output_folder, hu_threshold)
