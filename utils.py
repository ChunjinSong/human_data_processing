import h5py, cv2
from collections.abc import Iterable
import numpy as np
from collections import namedtuple
import math
import yaml
import os
from smplx.body_models import SMPL, SMPLX

def parse_config(yaml_file):
    config = None
    with open(yaml_file, 'r') as file:
        config = yaml.full_load(file)
    return config

def make_dirs(path):
    folder_path = path
    if os.path.splitext(path)[1]:
        folder_path = os.path.dirname(path)
    os.makedirs(folder_path, exist_ok=True)
    return path

def get_smpl(gender='neutral', model_path='../model/smpl_model_data'):

    smpl_model = SMPL(model_path, gender=gender).to( 'cuda:0')
    return smpl_model

#################################
#         Skeleton Helpers      #
#################################

Skeleton = namedtuple("Skeleton", ["joint_names", "joint_trees", "root_id", "nonroot_id", "cutoffs", "end_effectors"])

SMPLSkeleton = Skeleton(
    joint_names=[
        # 0-3
        'pelvis', 'left_hip', 'right_hip', 'spine1',
        # 4-7
        'left_knee', 'right_knee', 'spine2', 'left_ankle',
        # 8-11
        'right_ankle', 'spine3', 'left_foot', 'right_foot',
        # 12-15
        'neck', 'left_collar', 'right_collar', 'head',
        # 16-19
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        # 20-23,
        'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ],
    joint_trees=np.array(
                [0, 0, 0, 0,
                 1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 9, 9, 12,
                 13, 14, 16, 17,
                 18, 19, 20, 21]),
    root_id=0,
    nonroot_id=[i for i in range(24) if i != 0],
    cutoffs={'hip': 200, 'spine': 300, 'knee': 70, 'ankle': 70, 'foot': 40, 'collar': 100,
            'neck': 100, 'head': 120, 'shoulder': 70, 'elbow': 70, 'wrist': 60, 'hand': 60},
    end_effectors=[10, 11, 15, 22, 23],
)

def get_mask(msk):
    msk = (msk != 0).astype(np.uint8)
    border = 5
    kernel = np.ones((border, border), np.uint8)
    sampling_msk = cv2.dilate(msk.copy(), kernel, iterations=3)
    dilated = cv2.dilate(msk.copy(), kernel)
    eroded = cv2.erode(msk.copy(), kernel)
    sampling_msk[(dilated - eroded) == 1] = 128
    return msk, sampling_msk[..., None]

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, np.linalg.inv(pose)


def write_to_h5py_npc(
    filename: str,
    data: dict,
    img_chunk_size: int = 64,
    compression: str = 'gzip'
):

    imgs = data['imgs']
    H, W = imgs.shape[1:3]

    redundants = ['index', 'img_path']
    img_to_chunk = ['imgs', 'bkgds', 'masks']
    img_to_keep_whole = ['sampling_masks']

    for r in redundants:
        if r in data:
            data.pop(r)

    chunk = (1, int(img_chunk_size**2),)
    whole = (1, H * W,)

    h5_file = h5py.File(filename, 'w')

    # store meta
    ds = h5_file.create_dataset('img_shape', (4,), np.int32)
    ds[:] = np.array([*imgs.shape])

    for k in data.keys():
        if not isinstance(data[k], Iterable):
            print(f'{k}: non-iterable')
            ds = h5_file.create_dataset(k, (), type(data[k]))
            ds[()] = data[k]
            continue

        d_shape = data[k].shape
        C = d_shape[-1]
        N = d_shape[0]
        if k in img_to_chunk or k in img_to_keep_whole:
            data_chunk = chunk + (C,) if k in img_to_chunk else whole + (C,)
            flatten_shape = (N, H * W, C)
            print(f'{k}: img to chunk in size {data_chunk}, flatten as {flatten_shape}')
            # flatten the image for faster indexing
            ds = h5_file.create_dataset(k, flatten_shape, data[k].dtype,
                                        chunks=data_chunk, compression=compression)
            for idx in range(N):
                ds[idx] = data[k][idx].reshape(*flatten_shape[1:])
            #ds[:] = data[k].reshape(*flatten_shape)
        elif k == 'img_paths':
            img_paths = data[k].astype('S')
            ds = h5_file.create_dataset(k, (len(img_paths),), img_paths.dtype)
            ds[:] = img_paths
        else:
            if np.issubdtype(data[k].dtype, np.floating):
                dtype = np.float32
            elif np.issubdtype(data[k].dtype, np.integer):
                dtype = np.int64
            elif np.issubdtype(data[k].dtype,  np.dtype('S')):
                dtype = np.dtype('S32')
            else:
                raise NotImplementedError('Unknown datatype for key {k}: {data[k].dtype}')

            ds = h5_file.create_dataset(k, data[k].shape, dtype,
                                        compression=compression)
            ds[:] = data[k][:]
            print(f'{k}: data to store as {dtype}')
        pass

    h5_file.close()



def write_to_h5py(filename, data, img_chunk_size=64,
                compression='gzip'):

    imgs = data['images']
    H, W = imgs.shape[1:3]

    redundants = ['index', 'img_path']
    img_to_chunk = []
    img_to_keep_whole = ['masks_samp', 'images', 'masks']

    for r in redundants:
        if r in data:
            data.pop(r)

    chunk = (1, int(img_chunk_size**2),)
    whole = (1, H * W,)

    h5_file = h5py.File(filename, 'w')

    # store meta
    ds = h5_file.create_dataset('img_shape', (4,), np.int32)
    ds[:] = np.array([*imgs.shape])

    for k in data.keys():
        if not isinstance(data[k], Iterable):
            print(f'{k}: non-iterable')
            ds = h5_file.create_dataset(k, (), type(data[k]))
            ds[()] = data[k]
            continue

        d_shape = data[k].shape
        C = d_shape[-1]
        N = d_shape[0]
        if k in img_to_chunk or k in img_to_keep_whole:
            data_chunk = chunk + (C,) if k in img_to_chunk else whole + (C,)
            flatten_shape = (N, H * W, C)
            print(f'{k}: img to chunk in size {data_chunk}, flatten as {flatten_shape}')
            # flatten the image for faster indexing
            ds = h5_file.create_dataset(k, flatten_shape, data[k].dtype,
                                        chunks=data_chunk, compression=compression)
            for idx in range(N):
                ds[idx] = data[k][idx].reshape(*flatten_shape[1:])
            #ds[:] = data[k].reshape(*flatten_shape)
        elif k == 'img_paths':
            img_paths = data[k].astype('S')
            ds = h5_file.create_dataset(k, (len(img_paths),), img_paths.dtype)
            ds[:] = img_paths
        else:
            if np.issubdtype(data[k].dtype, np.floating):
                dtype = np.float32
            elif np.issubdtype(data[k].dtype, np.integer):
                dtype = np.int64
            elif np.issubdtype(data[k].dtype,  np.dtype('S')):
                dtype = np.dtype('S32')
            else:
                raise NotImplementedError('Unknown datatype for key {k}: {data[k].dtype}')

            ds = h5_file.create_dataset(k, data[k].shape, dtype,
                                        compression=compression)
            ds[:] = data[k][:]
            print(f'{k}: data to store as {dtype}')
        pass

    h5_file.close()



def get_kp_bounding_cylinder(kp, skel_type=SMPLSkeleton, ext_scale=0.001,
                             extend_mm=250, top_expand_ratio=1.,
                             bot_expand_ratio=0.25, head=None, verbose=False):
    '''
    head: -y for most dataset (SPIN estimated), z for SURREAL
    '''

    # g_axes: axes that define the ground plane
    # h_axis: axis that is perpendicular to the ground
    # flip: to flip the height (if the sky is on the negative part)
    assert head is not None, 'need to specify the direction of ground plane (i.e., the direction when the person stand up straight)!'
    if verbose:
        print(f'Head direction: {head}')
    if head.endswith('z'):
        g_axes = [0, 1]
        h_axis = 2
    elif head.endswith('y'):
        g_axes = [0, 2]
        h_axis = 1
    else:
        raise NotImplementedError(f'Head orientation {head} is not implemented!')
    flip = 1 if not head.startswith('-') else -1

    if skel_type is None:
        skel_type = SMPLSkeleton

    n_dim = len(kp.shape)
    # find root location
    root_id = skel_type.root_id
    if not isinstance(root_id, int):
        root_id = root_id[0] # use the first root
    root_loc = kp[..., root_id, :]

    # calculate distance to center line
    if n_dim == 2:
        dist = np.linalg.norm(kp[:, g_axes] - root_loc[g_axes], axis=-1)
    elif n_dim == 3: # batch
        dist = np.linalg.norm(kp[..., g_axes] - root_loc[:, None, g_axes], axis=-1)
        max_height = (flip * kp[..., h_axis]).max()
        min_height = (flip * kp[..., h_axis]).min()

    # find the maximum distance to center line (in mm*ext_scale)
    max_dist = dist.max(-1)
    max_height = (flip * kp[..., h_axis]).max(-1)
    min_height = (flip * kp[..., h_axis]).min(-1)

    # set the radius of cylinder to be a bit larger
    # so that every part of the human is covered
    extension = extend_mm * ext_scale
    radius = max_dist + extension
    top = flip * (max_height + extension * top_expand_ratio) # extend head a bit more
    bot = flip * (min_height - extension * bot_expand_ratio) # don't need that much for foot
    cylinder_params = np.stack([root_loc[..., g_axes[0]], root_loc[..., g_axes[1]],
                               radius, top, bot], axis=-1)
    return cylinder_params


def get_smpl_l2ws(pose, rest_pose=None, scale=1., skel_type=SMPLSkeleton, coord="xxx"):

    def mat_to_homo(mat):
        last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.concatenate([mat, last_row], axis=0)

    joint_trees = skel_type.joint_trees

    # apply scale
    rest_kp = rest_pose * scale
    mrots = [cv2.Rodrigues(p)[0] for p in pose]
    mrots = np.array(mrots)

    l2ws = []
    # local-to-world transformation
    l2ws.append(mat_to_homo(np.concatenate([mrots[0], rest_kp[0, :, None]], axis=-1)))
    mrots = mrots[1:]
    for i in range(rest_kp.shape[0] - 1):
        idx = i + 1
        # rotation relative to parent
        joint_rot = mrots[idx-1]
        joint_j = rest_kp[idx][:, None]

        parent = joint_trees[idx]
        parent_j = rest_kp[parent][:, None]

        # transfer from local to parent coord
        joint_rel_transform = mat_to_homo(
            np.concatenate([joint_rot, joint_j - parent_j], axis=-1)
        )

        # calculate kinematic chain by applying parent transform (to global)
        l2ws.append(l2ws[parent] @ joint_rel_transform)

    l2ws = np.array(l2ws)

    return l2ws

def rotate_x(phi):
    cos = np.cos(phi)
    sin = np.sin(phi)
    return np.array([[1,   0,    0, 0],
                     [0, cos, -sin, 0],
                     [0, sin,  cos, 0],
                     [0,   0,    0, 1]], dtype=np.float32)

def rotate_z(psi):
    cos = np.cos(psi)
    sin = np.sin(psi)
    return np.array([[cos, -sin, 0, 0],
                     [sin,  cos, 0, 0],
                     [0,      0, 1, 0],
                     [0,      0, 0, 1]], dtype=np.float32)
def rotate_y(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos,   0,  sin, 0],
                     [0,     1,    0, 0],
                     [-sin,  0,  cos, 0],
                     [0,   0,      0, 1]], dtype=np.float32)

def get_camera_motion_bullet(c2w, n_bullet=30, axis='y'):
    if axis == 'y':
        rotate_fn = rotate_y
    elif axis == 'x':
        rotate_fn = rotate_x
    elif axis == 'z':
        rotate_fn = rotate_z
    else:
        raise NotImplementedError(f'rotate axis {axis} is not defined')

    y_angles = np.linspace(0, math.radians(360), n_bullet + 1)[:-1]
    c2ws = []
    for a in y_angles:
        c = rotate_fn(a) @ c2w
        c2ws.append(c)
    return np.array(c2ws)


def rotate_vector(v, axis, n_bullet=30):
    """
    Rotate vector `v` around `axis` by `angle` (in radians).

    :param v: The vector to rotate.
    :param axis: The axis to rotate around.
    :param angle: The angle to rotate by in radians.
    :return: The rotated vector.
    """

    angles = np.linspace(0, math.radians(360), n_bullet + 1)[:-1]
    c2ws = []
    for angle in angles:
        axis = axis / np.linalg.norm(axis)  # Normalize the axis vector
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Rodrigues' rotation formula
        v_rot = (v * cos_angle +
                 np.cross(axis, v) * sin_angle +
                 axis * np.dot(axis, v) * (1 - cos_angle))
        c2ws.append(v_rot)

    return c2ws



