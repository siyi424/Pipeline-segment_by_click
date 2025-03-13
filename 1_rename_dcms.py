import os
import shutil
import pydicom
from tqdm import tqdm
from config import post_contrast_path, pre_contrast_path, input_post_dcms, input_pre_dcms

def rename_dicom_files(source_folder, target_folder):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(source_folder) if f.endswith('.dcm')]
    
    # 创建一个列表来存储文件路径和对应的序列号
    dicom_files = []
    
    for file in files:
        file_path = os.path.join(source_folder, file)
        # 读取DICOM文件头信息
        ds = pydicom.dcmread(file_path)
        # 获取序列号
        sequence_number = int(ds.InstanceNumber)
        dicom_files.append((file_path, sequence_number))
    
    # 按照序列号排序
    dicom_files.sort(key=lambda x: x[1])
    
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)
    
    # 复制并重新命名文件
    for index, (file_path, _) in enumerate(tqdm(dicom_files, desc="Copying and renaming files")):
        new_file_name = f"{index}.dcm"
        new_file_path = os.path.join(target_folder, new_file_name)
        shutil.copy2(file_path, new_file_path)

if __name__ == "__main__":
    # post_contrast
    source_folder = post_contrast_path
    target_folder = input_post_dcms

    os.makedirs(target_folder, exist_ok=True)
    rename_dicom_files(source_folder, target_folder)
    

    # pre_contrast
    source_folder = pre_contrast_path
    target_folder = input_pre_dcms
    os.makedirs(target_folder, exist_ok=True)
    rename_dicom_files(source_folder, target_folder)