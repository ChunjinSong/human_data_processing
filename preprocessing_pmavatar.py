import cv2
import os, h5py
import numpy as np
import argparse
import utils
from smplx.body_models import SMPL

device = 'cuda:0'

# camera coordinate transformation
to_npc_cam = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def main(args):
    smpl_model = SMPL('../model/smpl_model_data', gender=args.gender).to(device)
    source_file = args.source_file
    target_file = args.target_file

    head, tail = os.path.split(target_file)
    os.makedirs(head, exist_ok=True)

    data = h5py.File(source_file, 'r')
    print(f'============ {tail}: npc: start ===============')
    ext_scale = 0.001

    betas, kp3d, bones, skts, vertices = [], [], [], [], []
    c2ws, focals, centers = [], [], []

    data_len, H, W, _ = data['img_shape'][:]
    # smpl_shape = data['smpl_shape'][:]
    tpose_joints = data['cl_joints'][:]

    for idx in range(data_len):

        K = data['cameras_K'][idx]
        E = data['cameras_E'][idx]

        c2w = np.linalg.inv(E)
        c2w = c2w @ to_npc_cam
        c2ws.append(c2w)
        focals.append([K[0, 0], K[1, 1]])
        centers.append(K[:2, 2])

        poses = data['meshes_pose'][idx].reshape(-1, 3)
        Rh = data['meshes_Rh'][idx]
        Th = data['meshes_Th'][idx]

        global_rot = cv2.Rodrigues(Rh)[0]
        RT = np.eye(4)
        RT[:3, :3] = global_rot
        RT[:3, 3:] = Th.reshape(-1, 3, 1)

        l2w = utils.get_smpl_l2ws(poses, rest_pose=tpose_joints)
        l2w = RT @ l2w

        skt = np.linalg.inv(l2w)
        poses[..., :3] = Rh

        # betas.append(smpl_shape)
        kp3d.append(l2w[..., :3, -1])
        bones.append(poses)
        skts.append(skt)

    kp3d = np.array(kp3d)
    cyls = utils.get_kp_bounding_cylinder(
        kp3d,
        ext_scale=ext_scale,
        extend_mm=250,
        top_expand_ratio=1.00,
        bot_expand_ratio=0.25,
        head='y'
    )
    bkgd_idxs = np.zeros(data_len).astype(int)
    kp_idxs = np.arange(0, data_len)
    cam_idxs = kp_idxs

    data = {
        'imgs': np.array(data['images'][:]).reshape(-1, H, W, 3),
        'bkgds': np.array(np.zeros((1, H, W, 3), dtype=np.uint8)),
        'bkgd_idxs': bkgd_idxs,
        'masks': np.array(data['masks'][:]).reshape(-1, H, W, 1),
        'sampling_masks': np.array(data['masks_samp'][:]).reshape(-1, H, W, 1),
        'c2ws': np.array(c2ws).astype(np.float32),
        'img_pose_indices': cam_idxs,
        'kp_idxs': np.array(kp_idxs),
        'centers': np.array(centers).astype(np.float32),
        'focals': np.array(focals).astype(np.float32),
        'kp3d': kp3d.astype(np.float32),
        'betas': np.array(betas).astype(np.float32),
        'bones': np.array(bones).astype(np.float32),
        'skts': np.array(skts).astype(np.float32),
        'cyls': np.array(cyls).astype(np.float32),
        'rest_pose': np.array(tpose_joints).astype(np.float32),
        'frames_name': data['frames_name'][:],

    }

    utils.write_to_h5py_npc(target_file, data)

    print(f'============ {tail}: npc: end ===============')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processing data for NPC on a sequence")
    parser.add_argument('--seq', type=str, default='sequence_name', help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument('--source_file', type=str, default='../data/data_for_humannerf/sequence_name_training.h5')
    parser.add_argument('--target_file', type=str, default='../data/data_for_npc/sequence_name_training.h5')
    parser.add_argument('--gender', type=str, default='neutral')

    args = parser.parse_args()
    args.source_dir = f'../data/data_for_humannerf/{args.seq}.h5'
    args.target_file = f'../data/data_for_npc/{args.seq}.h5'
    main(args)
