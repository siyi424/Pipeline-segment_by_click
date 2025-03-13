import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import shutil

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

from sam2.build_sam import build_sam2_video_predictor
from config import sam2_checkpoint, model_cfg, it_num, tmp_frames_it_dir, seg_pre, seg_post, post_jpgs, pre_jpgs


# 手动更改
saved_path = seg_post
os.makedirs(saved_path, exist_ok=True)
video_dir = post_jpgs


# 构建SAM2视频预测器
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

############################################

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


############################################
frame_names = [
    p for p in os.listdir(video_dir)
    # sam2对视频/序列的分割仅支持.mp4\.jpg
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"] 
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


skip_all_rest = False

def save_black_masks_for_frames(frame_list, src_dir, dst_dir):
    # 保存传入 frame_list 中所有图片的全黑掩码
    for f_name in frame_list:
        img_path = os.path.join(src_dir, f_name)
        img = Image.open(img_path)
        black_mask = np.zeros((img.height, img.width), dtype=np.uint8)
        mask_image = Image.fromarray(black_mask)
        mask_path = f'{dst_dir}/{f_name}'
        mask_image.save(mask_path)
        print(f"Saved black mask for frame {f_name}")

############################################

for batch_id in range(len(frame_names) // it_num + 1):
    if skip_all_rest:
        break
    print(f"Batch_id: {batch_id}---------------------")

    # 将该组的图片复制到临时文件夹
    if len(frame_names) >= (batch_id + 1) * it_num:
        copy_frame_names = frame_names[batch_id * it_num: (batch_id + 1) * it_num]
    else:
        copy_frame_names = frame_names[batch_id * it_num:]
    print(f"Copying frames {copy_frame_names[0]} - {copy_frame_names[-1]}")

    # 将copy_frame_names中的图片复制到临时文件夹
    # 如果临时文件夹已经存在，则删除
    if os.path.exists(tmp_frames_it_dir):
        shutil.rmtree(tmp_frames_it_dir)
    os.makedirs(tmp_frames_it_dir)

    for frame_name in copy_frame_names:
        frame_path = os.path.join(video_dir, frame_name)
        tmp_frame_path = os.path.join(tmp_frames_it_dir, frame_name)
        shutil.copyfile(frame_path, tmp_frame_path)

    inference_state = predictor.init_state(video_path=tmp_frames_it_dir)
      
    # 初始化存储点击点的列表
    points = []
    labels = []

    # 定义点击事件的回调函数
    def onclick(event):
        global points, labels
        if event.button == 1:  # 左键：添加正点击
            points.append([int(event.xdata), int(event.ydata)])
            labels.append(1)
            plt.scatter(event.xdata, event.ydata, c='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
        elif event.button == 3:  # 右键：添加负点击
            points.append([int(event.xdata), int(event.ydata)])
            labels.append(0)
            plt.scatter(event.xdata, event.ydata, c='red', marker='*', s=200, edgecolor='white', linewidth=1.25)
        plt.draw()

    def on_key(event):
        if event.key == 'enter':
            plt.close()  # 按下回车键关闭图像窗口

    def get_user_points(image):
        global points, labels
        points = []
        labels = []
        fig, ax = plt.subplots()
        ax.imshow(np.array(image), cmap='gray')
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        kid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        fig.canvas.mpl_disconnect(kid)
        return np.array(points, dtype=np.int32), np.array(labels, dtype=np.int32)

    def get_segmentation_result(image, points, labels):
        if len(points) == 0:
            print("No points selected, skipping segmentation.")
            return None
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        fig, ax = plt.subplots()
        ax.imshow(np.array(image), cmap='gray')
        show_points(points, labels, ax)
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])
        plt.show()

        return out_mask_logits[0].cpu().numpy()

    frame_name = copy_frame_names[0]
    image = Image.open(os.path.join(video_dir, frame_name))

    while True:
        points, labels = get_user_points(image)
        if len(points) == 0:
            print("No points selected. Saving black masks for this batch and all subsequent batches...")
            # 本批次图片
            save_black_masks_for_frames(copy_frame_names, video_dir, saved_path)
            # 后续所有批次图片
            for next_batch_id in range(batch_id + 1, len(frame_names) // it_num + 1):
                if len(frame_names) >= (next_batch_id + 1) * it_num:
                    leftover = frame_names[next_batch_id * it_num : (next_batch_id + 1) * it_num]
                else:
                    leftover = frame_names[next_batch_id * it_num :]
                save_black_masks_for_frames(leftover, video_dir, saved_path)
            skip_all_rest = True
            break
        mask = get_segmentation_result(image, points, labels)
        if mask is None:
            continue
        user_input = input("Are you satisfied with the segmentation? (y/n): ")
        if user_input.lower() == 'y':
            break

    if skip_all_rest:
        continue

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }


    # 保存每帧的掩码
    for out_frame_idx, frame_name in enumerate(copy_frame_names):
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                # 调试信息：打印掩码的形状和数据类型
                print(f"Frame {out_frame_idx}, Object {out_obj_id}, Mask shape: {out_mask.shape}, dtype: {out_mask.dtype}")
                
                # 确保掩码是二维数组
                squeezed_mask = np.squeeze(out_mask)
                if squeezed_mask.ndim == 2:
                    mask_image = Image.fromarray((squeezed_mask * 255).astype('uint8'))
                    mask_path = f'{saved_path}/{frame_name}'
                    mask_image.save(mask_path)
                    print(f"Saved mask for frame {frame_name}, object {out_obj_id} to {mask_path}")
                else:
                    print(f"Error: Mask for frame {out_frame_idx}, object {out_obj_id} is not a 2D array.")
        else:
            print(f"Warning: No segmentation data for frame {out_frame_idx}")



