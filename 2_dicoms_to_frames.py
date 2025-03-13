import pydicom
import cv2
import os
from tqdm import tqdm
import numpy as np
from config import input_post_dcms, input_pre_dcms, post_jpgs, pre_jpgs, level, window

def apply_window_level(ds, level, window):
    """
    应用窗宽窗位到图像。

    参数:
    ds (pydicom.dataset.FileDataset): DICOM数据集。
    level (int): 窗位（Window Level）。
    window (int): 窗宽（Window Width）。

    返回:
    numpy.ndarray: 应用窗宽窗位后的图像。
    """
    intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0.0
    slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1.0

    img = ds.pixel_array * slope + intercept
    img_min = level - (window / 2)
    img_max = level + (window / 2)
    img_clipped = np.clip(img, img_min, img_max)
    img_normalized = ((img_clipped - img_min) / (img_max - img_min)) * 255.0
    return img_normalized.astype('uint8')

def dicom_to_jpg(dicom_folder, output_folder, level=1000, window=5000):
    """
    将DICOM序列图转换为png格式图片，并按顺序命名。

    参数:
    dicom_folder (str): 存放DICOM图像文件的文件夹路径。
    output_folder (str): 输出JPEG图片的文件夹路径。
    level (int): 窗位（Window Level）。
    window (int): 窗宽（Window Width）。
    """
    os.makedirs(output_folder, exist_ok=True)

    file_list = sorted(os.listdir(dicom_folder))
    for idx, filename in enumerate(tqdm(file_list, desc="Converting DICOM to JPG")):
        if filename.endswith(".dcm"):
            dicom_path = os.path.join(dicom_folder, filename)
            try:
                ds = pydicom.dcmread(dicom_path)
                img_windowed = apply_window_level(ds, level, window)
                new_filename = filename.replace(".dcm", ".jpg")
                output_path = os.path.join(output_folder, new_filename)
                cv2.imwrite(output_path, img_windowed)
            except Exception as e:
                print(f"Error reading {dicom_path}: {e}")


if __name__ == "__main__":
    # post contrast
    input_dcms = input_post_dcms
    output_jpgs = post_jpgs
    dicom_to_jpg(input_dcms, output_jpgs, level, window)

    # pre contrast
    input_dcms = input_pre_dcms
    output_jpgs = pre_jpgs
    dicom_to_jpg(input_dcms, output_jpgs, level, window)


