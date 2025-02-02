import os
import cv2
from shutil import copyfile
import torch
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
from absl import app
from absl import flags
from smplx.body_models import SMPL
from preprocessing_utils import transform_smpl
from normalize_cameras import normalize_cameras

FLAGS = flags.FLAGS

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

    subject_dir = os.path.join(dataset_dir, f"{subject}")
    smpl_params_dir = os.path.join(subject_dir, "new_params")
    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()
    cams = annots['cams']

    select_view = cfg_data['view']
    if len(select_view) == 0:
        select_view = [i for i in range(len(cams['K'])) if i not in cfg['training']['view']]


    # load cameras
    cam_Ks = np.array(cams['K'])[select_view].astype('float32')  # view_num*3*3
    cam_Rs = np.array(cams['R'])[select_view].astype('float32')  # view_num*3*3
    cam_Ts = np.array(cams['T'])[select_view].astype('float32') / 1000.  # view_num*3*1
    cam_Ds = np.array(cams['D'])[select_view].astype('float32')  # view_num*5*1

    K = np.zeros((cam_Ks.shape[0], 4, 4)).astype('float32')  # view_num*4*4
    K[:, :3, :3] = cam_Ks
    K[:, 3, 3] = 1.  # view_num*4*4

    E = np.zeros((cam_Ks.shape[0], 4, 4)).astype('float32')  # view_num*4*4
    E[:, :3, :3] = cam_Rs
    cam_T = cam_Ts[:, :3, 0]
    E[:, :3, 3] = cam_T
    E[:, 3, 3] = 1.  # view_num*4*4

    # load image paths
    img_path_frames_views = annots['ims']
    img_paths = np.array([
        np.array(multi_view_paths['ims'])[select_view] \
        for multi_view_paths in img_path_frames_views
    ])

    img_paths = img_paths[start_frame:end_frame][::skip]

    path_name = subject if 'name' not in cfg['output'].keys() else cfg['output']['name']
    path_name = path_name + '_' + cfg_data['name']
    output_path = os.path.join(cfg['output']['dir'], path_name)

    os.makedirs(output_path, exist_ok=True)
    out_img_dir = prepare_dir(output_path, 'image')
    out_mask_dir = prepare_dir(output_path, 'mask')

    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))


    all_betas = []
    output_trans, output_pose, output_P = [], [], {}
    idx_out = -1

    device = 'cuda:0'
    smpl_model = SMPL('../model/smpl_model_data', gender=gender).to(device)

    for idx_frame, path_frame in enumerate(tqdm(img_paths)):
        for idx_camera, path_camera in enumerate(path_frame):
            idx_out = idx_out + 1
            real_idx_frame = start_frame + idx_frame * skip
            img_path = os.path.join(subject_dir, path_camera)
            msk_path = os.path.join(subject_dir, 'mask/' + path_camera.replace('jpg', 'png'))

            print(img_path)

            # load image
            img = cv2.imread(img_path)
            img = cv2.undistort(img, cam_Ks[idx_camera], cam_Ds[idx_camera])
            cv2.imwrite(os.path.join(out_img_dir, '%04d.png' % idx_out), img)

            # load and write mask
            mask = cv2.imread(msk_path)
            mask = cv2.undistort(mask, cam_Ks[idx_camera], cam_Ds[idx_camera])
            cv2.imwrite(os.path.join(out_mask_dir, '%04d.png' % idx_out), mask)

            if subject in ['313', '315']:
                smpl_idx = real_idx_frame + 1  # index begin with 1
            else:
                smpl_idx = real_idx_frame

            # load smpl parameters
            smpl_params = np.load(
                os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
                allow_pickle=True).item()

            smpl_shape = smpl_params['shapes'][0]  # (10,)
            poses = smpl_params['poses'][0]  # (72,)
            Rh = smpl_params['Rh'][0]  # (3,)
            Th = smpl_params['Th'][0]  # (3,)
            global_rot = cv2.Rodrigues(Rh)[0]


            all_betas.append(smpl_shape)

            cam_extrinsics = E[idx_camera]
            smpl_pose = poses.reshape(-1)
            smpl_pose[:3] = Rh

            T_hip = smpl_model.get_T_hip(
                betas=torch.tensor(smpl_shape)[None].float().to(device)).squeeze().cpu().numpy()

            T_hip_trans = T_hip - np.sum(T_hip[..., np.newaxis, :] * global_rot, -1)
            smpl_trans = Th - T_hip_trans

            # transform the spaces such that our camera has the same orientation as the OpenGL camera
            target_extrinsic = np.eye(4)
            target_extrinsic[1:3] *= -1
            target_extrinsic, smpl_pose, smpl_trans = transform_smpl(cam_extrinsics, target_extrinsic, smpl_pose,
                                                                     smpl_trans, T_hip)

            smpl_output = smpl_model(betas=torch.tensor(smpl_shape)[None].float().to(device),
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

            target_extrinsic[0, 3] = target_extrinsic[0, 3] #* -1
            P = K[idx_camera] @ target_extrinsic
            output_trans.append(trans)
            output_pose.append(smpl_pose)

            output_P[f"cam_{idx_out}"] = P

    all_betas = np.array(all_betas)
    avg_betas = all_betas.mean(0)

    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'mean_shape.npy'), avg_betas)
    np.save(os.path.join(output_path, 'poses.npy'), np.array(output_pose))
    np.save(os.path.join(output_path, 'normalize_trans.npy'), np.array(output_trans))
    np.savez(os.path.join(output_path, "cameras.npz"), **output_P)

    normalize_cameras(os.path.join(output_path, "cameras.npz"), os.path.join(output_path, "cameras_normalize.npz"), num_of_cameras=-1)

def main(argv):
    del argv  # Unused.
    print(FLAGS.cfg)
    cfg = parse_config()

    process_data(cfg, 'training')
    process_data(cfg, 'novel_view')
    process_data(cfg, 'novel_pose')


flags.DEFINE_string('cfg',
                    'mocap/config/313.yaml',
                    'the path of config file')

if __name__ == '__main__':
    app.run(main)

