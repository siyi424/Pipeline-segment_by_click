
# ####################1_rename_dcms.py#######################
root_path = "xxxxx"
post_contrast_path = root_path + "post-contrast/"
pre_contrast_path = root_path + "pre-contrast/"
input_post_dcms = root_path + "tmp-contrast-filled/input_renamed/post_dcms/"
input_pre_dcms = root_path + "tmp-contrast-filled/input_renamed/pre_dcms/"


#######################2_dicoms_to_frames.py#######################
post_jpgs = root_path + "tmp-contrast-filled/input_jpgs/post_contrast/"
pre_jpgs = root_path + "tmp-contrast-filled/input_jpgs/pre_contrast/"
level = 1000
window = 5000

#######################3_seg_by_click#######################
sam2_checkpoint = "STEP1-segment_sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda"  # 或者 "cpu"
tmp_frames_it_dir = root_path + "tmp-contrast-filled/tmp_frames_it/"

# 该次标注用于分割多少张图片
it_num = 50

seg_pre = root_path + "tmp-contrast-filled/seg_masks/pre_contrast/"
seg_post = root_path + "tmp-contrast-filled/seg_masks/post_contrast/"


