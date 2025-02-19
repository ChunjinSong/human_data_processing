import os, pickle, glob, argparse, sys
import torch
import pytorch3d.transforms
from utils import *
from mvhuman.tools.camera_utils import read_camera_mvhumannet
from normalize_cameras import normalize_cameras
from preprocessing_utils import transform_smpl

device = 'cuda:0'

class Data_Process():
    def __init__(self, cfg):
        self.cfg = cfg
        self.gender = cfg['dataset']['gender']
        self.subject = cfg['dataset']['subject']
        self.source_dir = os.path.join(cfg['dataset']['path'], cfg['dataset']['subject'])
        self.target_dir = make_dirs(os.path.join(cfg['dataset']['path_out'], cfg['dataset']['subject']))
        self.lsavatar_dir = make_dirs(os.path.join(cfg['dataset']['path_out'], 'lsavatar_data'))

        self.source_smpl_dir = os.path.join(self.source_dir, 'smpl_neutral')
        self.smpl_model = get_smpl(gender=self.gender).to(device)

        extri_name = os.path.join(self.source_dir, 'camera_extrinsics.json')
        intri_name = os.path.join(self.source_dir, 'camera_intrinsics.json')
        cam_scale_name = os.path.join(self.source_dir, 'camera_scale.pkl')

        image_scale_factor = self.cfg['dataset']['image_scale_factor']
        self.cameras = read_camera_mvhumannet(intri_name, extri_name, cam_scale_name, image_scale_factor)

    def get_smpl(self, frame_idx):
        smpl_params = {}
        smpl_path = os.path.join(self.source_smpl_dir, f'{frame_idx:06}.pkl')
        smpl_param = pickle.load(open(smpl_path, 'rb'))
        betas = smpl_param['betas'].detach().squeeze().cpu().numpy()
        full_pose = pytorch3d.transforms.matrix_to_axis_angle(smpl_param['full_pose'])
        smpl_pose = full_pose.detach().squeeze().cpu().numpy().reshape(-1)
        smpl_trans = smpl_param['transl'].detach().squeeze().cpu().numpy()

        smpl_params['betas'] = betas
        smpl_params['smpl_pose'] = smpl_pose
        smpl_params['smpl_trans'] = smpl_trans

        return smpl_params

    def make_h5py(self, source_dir, target_file, data_type):
        make_dirs(target_file)
        # Read frames on directory
        img_dir = os.path.join(source_dir, 'images')
        msk_dir = os.path.join(source_dir, 'masks')

        imgPaths = sorted(glob.glob(f'{img_dir}/*'))
        maskPaths = sorted(glob.glob(f'{msk_dir}/*'))

        cameras = np.load(f'{source_dir}/cameras_normalize.npz')
        mean_shape = np.load(f'{source_dir}/mean_shape.npy')
        normalize_trans = np.load(f'{source_dir}/normalize_trans.npy')
        poses = np.load(f'{source_dir}/poses.npy')

        print(f'============ start: v2a.h5py ===============')
        images, masks = [], []
        frames_name = []
        cameras_out = []
        scales = []
        joints = []

        for idx, img_path in enumerate(imgPaths):
            oriImg = cv2.imread(os.path.join(img_path))
            oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

            mskImg = cv2.imread(os.path.join(maskPaths[idx]))[:, :, :1]
            mskImg = (mskImg > 127).astype(np.uint8)  # the value is 0 or 1

            images.append(oriImg)
            masks.append(mskImg)

            cameras_out.append(cameras[f'world_mat_{idx}'])
            scales.append(cameras[f'scale_mat_{idx}'])

            frames_name.append(os.path.basename(img_path))

            smpl_pose = poses[idx].copy()
            trans = normalize_trans[idx]
            smpl_output = self.smpl_model(betas=torch.tensor(mean_shape)[None].float().to(device),
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
            'cameras': np.array(cameras_out),
            'scales': np.array(scales),

            'mean_shape': np.array(mean_shape),
            'normalize_trans': np.array(normalize_trans),
            'poses': np.array(poses),
            'joints': np.array(joints),
            # 'rest_pose': np.array(tpose_joints).astype(np.float32), # do not use
        }

        write_to_h5py(target_file, data)
        print(f'============ {self.subject}_{data_type}.h5 end ===============')

    def process_data(self, data_type):
        print(f'====== Start ====== {self.subject}, {data_type} ===============')
        cfg_data = self.cfg[data_type]
        start_frame = cfg_data['start_frame']
        end_frame = cfg_data['end_frame']
        skip = cfg_data['skip']
        views = cfg_data['view']
        image_size = self.cameras['img_size']

        source_img_dir = os.path.join(self.source_dir, 'images_lr')
        # source_msk_dir = os.path.join(self.source_dir, 'masks_som')
        source_msk_dir = os.path.join(self.source_dir, 'fmask_lr')

        target_dir = self.target_dir + '_' + data_type

        target_img_dir = make_dirs(os.path.join(target_dir, 'images'))
        target_msk_dir = make_dirs(os.path.join(target_dir, 'masks'))

        idx_out = -1
        smpl_shapes = []
        output_trans, output_pose, output_P = [], [], {}
        for cam_idx in views:
            cam_idx = str(cam_idx)
            source_img_cam_dir = os.path.join(source_img_dir, cam_idx)
            source_msk_cam_dir = os.path.join(source_msk_dir, cam_idx)

            img_files = sorted(os.listdir(source_img_cam_dir))
            msk_files = sorted(os.listdir(source_msk_cam_dir))

            if end_frame != -1:
                frame_idxs = range(start_frame, end_frame, skip)
            else:
                frame_idxs = range(start_frame, len(img_files), skip)

            for frame_idx in frame_idxs:
                idx_out += 1

                source_msk = os.path.join(source_msk_cam_dir, msk_files[frame_idx])
                target_msk = os.path.join(target_msk_dir, f'{idx_out:04}.png')
                mask_data = cv2.resize(cv2.imread(source_msk, cv2.IMREAD_GRAYSCALE), [image_size[0], image_size[1]])
                mask_data = (mask_data[...] > 127).astype(np.uint8) * 255 # the value is 0 or 1
                cv2.imwrite(target_msk, mask_data)

                source_img = os.path.join(source_img_cam_dir, img_files[frame_idx])
                target_img = os.path.join(target_img_dir, f'{idx_out:04}.jpg')
                image_data = cv2.resize(cv2.imread(source_img), [image_size[0], image_size[1]])

                msk = (mask_data[...,None] > 127).astype(np.uint8)  # the value is 0 or 1
                image_data = (msk * image_data + (1.0 - msk) * 255).astype(np.uint8)

                cv2.imwrite(target_img, image_data)

                pose_id = int(img_files[frame_idx].split('_')[0])
                pose_id = int(pose_id / 5 - 1)
                smpl_params = self.get_smpl(pose_id)
                smpl_shapes.append(smpl_params['betas'])

                T_hip = self.smpl_model.get_T_hip(betas=torch.tensor(smpl_params['betas'])[None].float().to(device)).squeeze().cpu().numpy()

                # transform the spaces such that our camera has the same orientation as the OpenGL camera
                cam_extrinsics = self.cameras[cam_idx]['E']
                target_extrinsic = np.eye(4)
                target_extrinsic[1:3] *= -1
                target_extrinsic, smpl_pose, smpl_trans = transform_smpl(cam_extrinsics, target_extrinsic, smpl_params['smpl_pose'], smpl_params['smpl_trans'], T_hip)

                smpl_output = self.smpl_model(betas=torch.tensor(smpl_params['betas'])[None].float().to(device),
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
                P = self.cameras[cam_idx]['K'] @ target_extrinsic
                output_trans.append(trans)
                output_pose.append(smpl_pose)

                output_P[f"cam_{idx_out}"] = P

        all_betas = np.array(smpl_shapes)
        avg_betas = all_betas.mean(0)

        np.save(os.path.join(target_dir, 'mean_shape.npy'), avg_betas)
        np.save(os.path.join(target_dir, 'poses.npy'), np.array(output_pose))
        np.save(os.path.join(target_dir, 'normalize_trans.npy'), np.array(output_trans))
        np.savez(os.path.join(target_dir, "cameras.npz"), **output_P)

        normalize_cameras(os.path.join(target_dir, "cameras.npz"), os.path.join(target_dir, "cameras_normalize.npz"),
                          num_of_cameras=-1)

        print(f'====== Finish ====== {self.subject}, {data_type} ===============')

        h5py_path = make_dirs(os.path.join(self.lsavatar_dir, f'{self.subject}_{data_type}.h5'))
        self.make_h5py(target_dir, h5py_path, data_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing data for v2a")
    parser.add_argument('--cfg', type=str, default='./mvhuman/config/100846.yaml')
    parser.add_argument('--is_sam', type=bool, default=True)
    parser.add_argument('--data_type', type=str, default='training')

    args = parser.parse_args()
    print(args.cfg, args.data_type, args.is_sam)
    cfg = parse_config(args.cfg)
    process_handeler = Data_Process(cfg)
    # if args.is_sam:
    #     process_handeler.segmentation(args.data_type)
    process_handeler.process_data(args.data_type)


