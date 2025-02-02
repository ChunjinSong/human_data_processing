import sys
import cv2
import os
import numpy as np
import argparse
import time
import glob
from openpose.src import torch_openpose,util
from sklearn.neighbors import NearestNeighbors

def get_bbox_center(_img, mask):
    W, H = _img.shape[1], _img.shape[0]

    mask = mask[:, :, 0]
    where = np.asarray(np.where(mask))
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)
    left, top, right, bottom = bbox_min[1], bbox_min[0], bbox_max[1], bbox_max[
        0]
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, W)
    bottom = min(bottom, H)
    bbox_center = np.array([left + (right - left) / 2, top + (bottom - top) / 2])
    return bbox_center


def main(args):
    body_estimation = torch_openpose.body_25('body_25')

    DIR = '../raw_data'
    # Read frames on directory
    img_dir = f'{DIR}/{args.seq}/frames'
    msk_dir = f'{DIR}/{args.seq}/init_mask'

    imgPaths = sorted([f for f_ in [glob.glob(e) for e in (f'{img_dir}/*.png', f'{img_dir}/*.jpg')] for f in f_])
    maskPaths = sorted(glob.glob(f'{msk_dir}/*.png'))

    start = time.time()

    if not os.path.exists(f'{img_dir}/../openpose'):
        os.makedirs(f'{img_dir}/../openpose')

    # Process and display images
    nbrs = NearestNeighbors(n_neighbors=1)

    for idx, img_path in enumerate(imgPaths):
        oriImg = cv2.imread(os.path.join(img_path))
        mskImg = cv2.imread(os.path.join(maskPaths[idx]))
        bbox_center = get_bbox_center(oriImg, mskImg)
        candidate, subset = body_estimation(oriImg)


        candidate = np.array(candidate)  # [0][:, :2].astype(int)
        nbrs.fit(candidate[:, 8, :2])
        actor = nbrs.kneighbors(bbox_center.reshape(1, -1), return_distance=False).ravel()[0]
        poseKeypoints = candidate[actor]

        outImg = util.draw_bodypose(oriImg, [poseKeypoints], 'body_25')

        np.save(f'{img_dir}/../openpose/%04d.npy' % idx, poseKeypoints)
        cv2.imwrite(f'{img_dir}/../openpose/%04d.png' % idx, outImg)
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    # except Exception as e:
    #     print(e)
    #     sys.exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run OpenPose on a sequence")
    # sequence name
    parser.add_argument('--seq', type=str, default='story_training', help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_args()
    main(args)