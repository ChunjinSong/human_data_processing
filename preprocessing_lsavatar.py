import torch
import cv2
import os
import numpy as np
import argparse
import glob
import utils
from smplx.body_models import SMPL

device = 'cuda:0'

def process(args):
    args.source_dir = f'./data/{args.seq}'
    args.target_file = f'./data/data_for_lsavatar/{args.seq}.h5'

    smpl_model = SMPL('./model/smpl_model_data', gender=args.gender).to(device)

    source_dir = args.source_dir
    target_file = args.target_file
    head, tail = os.path.split(target_file)
    os.makedirs(head, exist_ok=True)
    # Read frames on directory
    img_dir = f'{source_dir}/image'
    msk_dir = f'{source_dir}/mask'

    imgPaths = sorted(glob.glob(f'{img_dir}/*.*'))
    maskPaths = sorted(glob.glob(f'{msk_dir}/*.*'))

    cameras = np.load(f'{source_dir}/cameras_normalize.npz')
    mean_shape = np.load(f'{source_dir}/mean_shape.npy')
    normalize_trans = np.load(f'{source_dir}/normalize_trans.npy')
    poses = np.load(f'{source_dir}/poses.npy')

    print(f'============ {tail}: lsavatar: start ===============')
    images, masks, masks_samp = [], [], []
    frames_name = []
    cameras_out = []
    scales = []
    joints = []

    T_hip = smpl_model.get_T_hip(betas=torch.tensor(mean_shape)[None].float().to(device)).squeeze().cpu().numpy()
    tpose_output = smpl_model(betas=torch.tensor(mean_shape)[None].float().to(device),
                              body_pose=torch.tensor(np.zeros_like(poses[0][3:]))[None].float().to(device),
                              joints_return_type='smpl')
    tpose_joints = tpose_output.joints.data.cpu().numpy().squeeze() - T_hip

    for idx, img_path in enumerate(imgPaths):
        oriImg = cv2.imread(os.path.join(img_path))
        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

        mskImg = cv2.imread(os.path.join(maskPaths[idx]))[:, :, :1]
        mskImg, sampling_mskImg = utils.get_mask(mskImg)

        images.append(oriImg)
        masks.append(mskImg)
        masks_samp.append(sampling_mskImg)

        cameras_out.append(cameras[f'world_mat_{idx}'])
        scales.append(cameras[f'scale_mat_{idx}'])

        frames_name.append(os.path.basename(img_path))

        smpl_pose = poses[idx].copy()
        trans = normalize_trans[idx]
        smpl_output = smpl_model(betas=torch.tensor(mean_shape)[None].float().to(device),
                                 body_pose=torch.tensor(smpl_pose[3:])[None].float().to(device),
                                 global_orient=torch.tensor(smpl_pose[:3])[None].float().to(device),
                                 transl=torch.tensor(trans)[None].float().to(device),
                                 joints_return_type='smpl')
        joint = smpl_output.joints.data.cpu().numpy().squeeze()
        joints.append(joint)

    data = {
        'frames_name': np.array(frames_name, dtype=np.dtype('S32')),
        'images': np.array(images),
        'masks': np.array(masks),
        'masks_samp': np.array(masks_samp), # not used
        'cameras': np.array(cameras_out), # camera intrinsic and extrinsic parameters
        'scales': np.array(scales), # the scaling caused by camera normalization

        'mean_shape': np.array(mean_shape), # the shape parameter of smpl
        'normalize_trans': np.array(normalize_trans), # the global translation
        'poses': np.array(poses), # the smpl pose parameters
        'joints': np.array(joints), # the location of joints 
        'rest_pose': np.array(tpose_joints).astype(np.float32), # the location of joints under T-pose
    }

    utils.write_to_h5py(target_file, data)
    print(f'============ {tail}: lsavatar: end ===============')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processing data for lsavatar on a sequence")
    parser.add_argument('--seq', type=str, default='sequence_name',help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument('--source_dir', type=str, default='./data/sequence_name_training')
    parser.add_argument('--target_file', type=str, default='./data/data_for_lsavatar/sequence_name_training.h5')
    parser.add_argument('--gender', type=str, default='neutral')

    args = parser.parse_args()
    args.source_dir = f'./data/{args.seq}'
    args.target_file = f'./data/data_for_lsavatar/{args.seq}.h5'
    process(args)

