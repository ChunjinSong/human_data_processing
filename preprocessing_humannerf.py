import torch
import cv2
import os, sys
import numpy as np
import argparse
import glob
import utils #import write_to_h5py
from smplx.body_models import SMPL

device = 'cuda:0'

def main(args):
    args.source_dir = f'./data/{args.seq}'
    args.target_file = f'./data/data_for_humannerf/{args.seq}.h5'
    smpl_model = SMPL('./model/smpl_model_data', gender=args.gender).to(device)

    source_dir = args.source_dir
    target_file = args.target_file

    head, tail = os.path.split(target_file)
    os.makedirs(head, exist_ok=True)

    # Read frames on directory
    img_dir = f'{source_dir}/image'
    msk_dir = f'{source_dir}/mask'

    imgPaths = sorted(glob.glob(f'{img_dir}/*.png'))
    maskPaths = sorted(glob.glob(f'{msk_dir}/*.png'))

    cameras = np.load(f'{source_dir}/cameras.npz')
    mean_shape = np.load(f'{source_dir}/mean_shape.npy')
    normalize_trans = np.load(f'{source_dir}/normalize_trans.npy')
    poses = np.load(f'{source_dir}/poses.npy')

    print(f'============ {tail}: humannerf: start ===============')
    images, masks, masks_samp = [], [], []
    frames_name = []
    cameras_K, cameras_E = [], []
    meshes_Rh, meshes_Th, meshes_pose, meshes_joints, meshes_tpose_joints = [], [], [], [], []

    T_hip = smpl_model.get_T_hip(betas=torch.tensor(mean_shape)[None].float().to(device)).squeeze().cpu().numpy()
    tpose_output = smpl_model(betas=torch.tensor(mean_shape)[None].float().to(device),
                              body_pose=torch.tensor(np.zeros_like(poses[0][3:]))[None].float().to(device),
                              joints_return_type='smpl')

    tpose_joints = tpose_output.joints.data.cpu().numpy().squeeze() - T_hip
    tpose_verts = tpose_output.vertices.data.cpu().numpy().squeeze() - T_hip

    for idx, img_path in enumerate(imgPaths):
        oriImg = cv2.imread(os.path.join(img_path))
        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

        mskImg = cv2.imread(os.path.join(maskPaths[idx]))[:, :, :1]
        mskImg, sampling_mskImg = utils.get_mask(mskImg)

        images.append(oriImg)
        masks.append(mskImg)
        masks_samp.append(sampling_mskImg)

        P = cameras[f'cam_{idx}'][:3, :4]
        K, E = utils.load_K_Rt_from_P(None, P)

        cameras_K.append(K)
        cameras_E.append(E)

        smpl_pose = poses[idx]
        smpl_trans = normalize_trans[idx]
        Rh = smpl_pose[:3].copy()
        smpl_pose[:3] = 0.

        smpl_output_woRT = smpl_model(betas=torch.tensor(mean_shape)[None].float().to(device),
                                      body_pose=torch.tensor(smpl_pose[3:])[None].float().to(device),
                                      joints_return_type='smpl')

        joints_woRT = smpl_output_woRT.joints.data.cpu().numpy().squeeze()
        joints = joints_woRT - joints_woRT[0]
        Th = smpl_trans + T_hip

        meshes_Rh.append(Rh)
        meshes_Th.append(Th)
        meshes_pose.append(smpl_pose)
        meshes_joints.append(joints)
        meshes_tpose_joints.append(tpose_joints)
        frames_name.append(os.path.basename(img_path))


    data = {
        'frames_name': np.array(frames_name, dtype=np.dtype('S32')),  #data['frames_name'][0].decode('UTF-8')

        'images': np.array(images),
        'masks': np.array(masks),
        'masks_samp': np.array(masks_samp),

        'cameras_K': np.array(cameras_K),
        'cameras_E': np.array(cameras_E),

        'meshes_Rh': np.array(meshes_Rh),
        'meshes_Th': np.array(meshes_Th),
        'meshes_pose': np.array(meshes_pose),
        'meshes_joints': np.array(meshes_joints),
        'meshes_tpose_joints': np.array(meshes_tpose_joints),

        'cl_joints': np.array(tpose_joints),
        'cl_verts': np.array(tpose_verts),
        'smpl_shape': np.array(mean_shape)
    }

    utils.write_to_h5py(target_file, data)
    print(f'============ {tail}: humannerf: end ===============')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processing data for humannerf on a sequence")
    # sequence name
    parser.add_argument('--seq', type=str, default='sequence_name', help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument('--source_dir', type=str, default='../data/gBR_sFM_c01_d04_mBR0_ch01_novel_pose')
    parser.add_argument('--target_file', type=str, default='../data/data_for_humannerf/aist/gBR_sFM_c01_d04_mBR0_ch01_novel_pose.h5')
    parser.add_argument('--gender', type=str, default='neutral')

    args = parser.parse_args()
    args.source_dir = f'../data/{args.seq}'
    args.target_file = f'../data/data_for_humannerf/{args.seq}.h5'

    main(args)
