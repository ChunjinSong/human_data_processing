import os
import cv2
import numpy as np


def collect_imgs_for_camera_select():
    actor = 'Actor01'
    seq = 'Sequence2'

    source_dir = f'/media/chunjins/My Passport/project/HumanNeRF/0_dataset/Actor-HQ/{actor}/{seq}/4x/rgbs'
    target_dir = f'/ubc/cs/home/c/chunjins/chunjin_scratch/project/humannerf/project/video_processing_4_humannerf/raw_data/actorshq/for_cam_select/{actor}/{seq}'
    os.makedirs(target_dir, exist_ok=True)
    scale_factor = 0.5
    img_skip = 30
    n_img = 12
    cam_dirs = sorted(os.listdir(source_dir))
    for cam_dir in cam_dirs:
        current_dir = os.path.join(source_dir, cam_dir)
        # current_target_dir = os.path.join(target_dir, cam_dir)
        # os.makedirs(current_target_dir, exist_ok=True)

        files_name = sorted(os.listdir(current_dir))[::img_skip][:n_img]
        imgs_out = []
        imgs_out_line = []
        for idx, file in enumerate(files_name):
            current_img = cv2.imread(os.path.join(current_dir, file))
            # Get original dimensions
            height, width = current_img.shape[:2]

            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize the image
            resized_image = cv2.resize(current_img, (new_width, new_height))
            imgs_out_line.append(resized_image)
            if idx > 0 and (idx+1) % 4 == 0:
                img_line = np.concatenate(imgs_out_line, axis=1)
                imgs_out.append(img_line)
                imgs_out_line = []
        img_out = np.concatenate(imgs_out, axis=0)
        cv2.imwrite(os.path.join(target_dir, f'{cam_dir}.jpg'), img_out)


def collect_imgs_single_camera():
    actor = 'Actor01'
    seq = 'Sequence2'
    camera = 126
    source_dir = f'/media/chunjins/My Passport/project/HumanNeRF/0_dataset/Actor-HQ/{actor}/{seq}/4x/rgbs'
    target_dir = f'/ubc/cs/home/c/chunjins/chunjin_scratch/project/humannerf/project/video_processing_4_humannerf/raw_data/actorshq/single_camera/{actor}/{seq}_{camera}'
    os.makedirs(target_dir, exist_ok=True)
    scale_factor = 0.5
    img_skip = 10
    cam_dirs = sorted(os.listdir(source_dir))

    cam_dirs = [cam_dirs[camera]]
    for cam_dir in cam_dirs:
        current_dir = os.path.join(source_dir, cam_dir)
        current_target_dir = os.path.join(target_dir, cam_dir)
        os.makedirs(current_target_dir, exist_ok=True)

        files_name = sorted(os.listdir(current_dir))[::img_skip]

        for idx, file in enumerate(files_name):
            current_img = cv2.imread(os.path.join(current_dir, file))
            # Get original dimensions
            height, width = current_img.shape[:2]

            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize the image
            resized_image = cv2.resize(current_img, (new_width, new_height))
            cv2.imwrite(os.path.join(current_target_dir, f'{idx}.jpg'), resized_image)


collect_imgs_single_camera()
