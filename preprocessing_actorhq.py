import os
import shutil
import pytorch3d.transforms
import pickle
import yaml
import torch
import numpy as np
from PIL import Image
from absl import app
from absl import flags
from smplx.body_models import SMPL
from actorhq.tools.camera_data import read_calibration_csv
from normalize_cameras import normalize_cameras

FLAGS = flags.FLAGS

device = 'cuda:0'

from preprocessing_utils import transform_smpl

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def to_3ch_image(image):
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        assert image.shape[2] == 1
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image

def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img

def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config

def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir

def process_data(cfg, type):
    subject = cfg['dataset']['subject']
    gender = cfg['dataset']['gender']
    dataset_dir = cfg['dataset']['path']

    cfg_data = cfg[type]
    start_frame = cfg_data['start_frame']
    end_frame = cfg_data['end_frame']
    skip = cfg_data['skip']
    views = cfg_data['view']

    smpl_model = SMPL('../model/smpl_model_data', gender=gender).to(device)

    path_cam = os.path.join(dataset_dir, 'calibration.csv')
    dir_img = os.path.join(dataset_dir, 'image')
    dir_msk = os.path.join(dataset_dir, 'mask')
    dir_smpl = os.path.join(dataset_dir, 'smpl')

    dir_out = os.path.join(cfg['output']['dir'], subject+'_'+type)
    dir_out_img = os.path.join(dir_out, 'image')
    dir_out_msk = os.path.join(dir_out, 'mask')

    os.makedirs(dir_out_img, exist_ok=True)
    os.makedirs(dir_out_msk, exist_ok=True)

    cameras = read_calibration_csv(path_cam)

    idx_out = -1
    all_betas = []
    output_trans, output_pose, output_P = [], [], {}
    for idx_cam in views:
        K3 = cameras[idx_cam].intrinsic_matrix()
        K = np.eye(4)
        K[:3, :3] = K3
        cam_extrinsics = np.linalg.inv(cameras[idx_cam].extrinsic_matrix_cam2world())


        dir_img_view = os.path.join(dir_img, f'Cam{idx_cam+1:03d}')
        dir_msk_view = os.path.join(dir_msk, f'Cam{idx_cam+1:03d}')

        for idx_frame in range(start_frame, end_frame, skip):
            name_img = f'Cam{idx_cam+1:03d}_rgb{idx_frame:06d}.jpg'
            name_msk = f'Cam{idx_cam+1:03d}_mask{idx_frame:06d}.png'
            name_frame = f'{idx_frame:04d}.pkl'
            idx_out += 1
            shutil.copy(os.path.join(dir_img_view, name_img), os.path.join(dir_out_img, f'{idx_out:04d}.jpg'))
            shutil.copy(os.path.join(dir_msk_view, name_msk), os.path.join(dir_out_msk, f'{idx_out:04d}.png'))

            with open(os.path.join(dir_smpl, name_frame), 'rb') as f:
                smpl = pickle.load(f)

            beta = smpl['betas'].detach().squeeze().cpu().numpy()
            full_pose = pytorch3d.transforms.matrix_to_axis_angle(smpl['full_pose'])
            smpl_pose = full_pose.detach().squeeze().cpu().numpy().reshape(-1)
            smpl_trans = smpl['transl'].detach().squeeze().cpu().numpy()

            all_betas.append(beta)

            T_hip = smpl_model.get_T_hip(
                betas=torch.tensor(beta)[None].float().to(device)).squeeze().cpu().numpy()

            # transform the spaces such that our camera has the same orientation as the OpenGL camera
            target_extrinsic = np.eye(4)
            target_extrinsic[1:3] *= -1
            target_extrinsic, smpl_pose, smpl_trans = transform_smpl(cam_extrinsics, target_extrinsic, smpl_pose,
                                                                     smpl_trans, T_hip)

            smpl_output = smpl_model(betas=torch.tensor(beta)[None].float().to(device),
                                     body_pose=torch.tensor(smpl_pose[3:])[None].float().to(device),
                                     global_orient=torch.tensor(smpl_pose[:3])[None].float().to(device),
                                     transl=torch.tensor(smpl_trans)[None].float().to(device))

            smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

            # we need to center the human for every frame due to the potentially large global movement
            v_max = smpl_verts.max(axis=0)
            v_min = smpl_verts.min(axis=0)
            normalize_shift = -(v_max + v_min) / 2.

            trans = smpl_trans + normalize_shift

            target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (target_extrinsic[:3, :3] @ normalize_shift)

            target_extrinsic[0, 3] = target_extrinsic[0, 3]  # * -1
            P = K @ target_extrinsic
            output_trans.append(trans)
            output_pose.append(smpl_pose)

            output_P[f"cam_{idx_out}"] = P

    all_betas = np.array(all_betas)
    avg_betas = all_betas.mean(0)

    np.save(os.path.join(dir_out, 'mean_shape.npy'), avg_betas)
    np.save(os.path.join(dir_out, 'poses.npy'), np.array(output_pose))
    np.save(os.path.join(dir_out, 'normalize_trans.npy'), np.array(output_trans))
    np.savez(os.path.join(dir_out, "cameras.npz"), **output_P)

    normalize_cameras(os.path.join(dir_out, "cameras.npz"), os.path.join(dir_out, "cameras_normalize.npz"), num_of_cameras=-1)

def main(argv):
    del argv  # Unused.
    cfg = parse_config()

    process_data(cfg, 'training')
    process_data(cfg, 'novel_view')
    process_data(cfg, 'novel_pose')


flags.DEFINE_string('cfg',
                    'actorhq/actor01_seq1.yaml',
                    'the path of config file')

if __name__ == '__main__':
    app.run(main)

