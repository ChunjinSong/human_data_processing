import torch
import cv2
import os, sys, time
import numpy as np
import argparse
import glob
from segment_anything import SamPredictor, sam_model_registry
sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def main(args):

    try:
        DIR = './data'
        # Read frames on directory
        img_dir = f'{DIR}/{args.seq}/image'
        msk_dir = f'{DIR}/{args.seq}/mask'

        imgPaths = sorted(glob.glob(f'{img_dir}/*.png'))
        maskPaths = sorted(glob.glob(f'{msk_dir}/*.png'))

        start = time.time()

        print(f'============ {args.seq}: sag mask: start ===============')
        for idx, img_path in enumerate(imgPaths):
            oriImg = cv2.imread(os.path.join(img_path))
            mskImg = cv2.imread(os.path.join(maskPaths[idx]))

            border = 20
            kernel = np.ones((border, border), np.uint8)
            mask = cv2.dilate(mskImg.copy(), kernel)[:,:,:1]

            x, y, w, h = cv2.boundingRect(mask)
            input_box = np.array([x, y, x + w, y + h])

            predictor.set_image(oriImg)
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            msk_sam = masks[0].astype(np.uint8)

            # msk_sam = cv2.cvtColor(msk_sam * 255, cv2.COLOR_GRAY2BGR)
            msk_sam = msk_sam * 255
            cv2.imwrite(os.path.join(maskPaths[idx]), msk_sam)

        end = time.time()
        print("SAM demo successfully finished. Total time: " + str(end - start) + " seconds")
        print(f'============ {args.seq}: sag mask: end ===============')

    except Exception as e:
        print(e)
        sys.exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run OpenPose on a sequence")
    # sequence name
    parser.add_argument('--seq', type=str, default='sequence_name', help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_args()
    main(args)