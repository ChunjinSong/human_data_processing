# Locality Sensitive Avatars From Video
## [Paper](https://openreview.net/pdf?id=SVta2eQNt3) | [Project Page]()


Data Preprocessing for ICLR 2025 paper [*LS_Avatar: Locality Sensitive Avatars From Video*](https://openreview.net/pdf?id=SVta2eQNt3) and baselines. 

* We provide the data processing code for [LS_Avatar](https://openreview.net/pdf?id=SVta2eQNt3) as well as for baselines, including [Vid2Avatar](https://github.com/MoyGcc/vid2avatar), [HumanNeRF](https://grail.cs.washington.edu/projects/humannerf/), [MonoHuman](https://yzmblog.github.io/projects/MonoHuman/), [NPC](https://lemonatsu.github.io/npc/), [PM-Avatar](https://github.com/ChunjinSong/pmavatar). However, please note that, except for Vid2Avatar, the processed data for other baselines is stored in an HDF5 (h5py) file. As a result, their dataloaders will need to be modified accordingly. I will release my implementation later.

## Getting Started
* Please follow [code for LS-Avatar](https://github.com/ChunjinSong/lsavatar) to set up the environment.
* Download [SMPL model](https://smpl.is.tue.mpg.de/download.php) (1.0.0 for Python 2.7 (10 shape PCs)) and move them to the corresponding places:
```
cd the/path/to/this/project
cd ../
mkdir model/smpl_model_data/
cd the/path/to/this/project
mv /path/to/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl ../model/smpl_model_data/SMPL_FEMALE.pkl
mv /path/to/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl ../model/smpl_model_data/SMPL_MALE.pkl
mv /path/to/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl ../model/smpl_model_data/SMPL_NEUTRAL.pkl
```

## Process ZJU-Mocap Dataset
We follow the protocol of [MonoHuman](https://github.com/Yzmblog/MonoHuman) to process ZJU-Mocap dataset.
* Download [ZJU-Mocap data](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset).
* Run the data processing code as below:
```
python preprocessing_mocap.py --cfg mocap/config/313.yaml  # this is to get the data for Vid2Avatar
python preprocessing_lsavatar.py --seq mocap/313_training  # this is to get data for LS_Avatar
python preprocessing_humannerf.py --seq mocap/313_training  # this is to get data for HumanNeRF and MonoHuman
python preprocessing_pmavatar.py --seq mocap/313_training  # this is to get data for NPC
```

## Process ActorHQ Dataset
* Download [ActorHQ data](https://actors-hq.com/)
* Run the data processing code as below:
```
python preprocessing_actorhq.py --cfg actorhq/config/actor01_seq1.yaml  # this is to get the data for Vid2Avatar
python preprocessing_lsavatar.py --seq actorhq/actor0101_training  # this is to get data for LS_Avatar
python preprocessing_humannerf.py --seq actorhq/actor0101_training  # this is to get data for HumanNeRF and MonoHuman
python preprocessing_pmavatar.py --seq actorhq/actor0101_training  # this is to get data for NPC
```

## Process MvHumanNet Dataset
* Download [MvHumanNet data](https://x-zhangyang.github.io/MVHumanNet/)
* MvHumanNet provides SMPL-X parameters, which need to be converted to SMPL parameters. Please follow the instructions of [SMPL-X](https://github.com/gngdb/smplx/tree/master/transfer_model) to perform the conversion and save the converted SMPL parameters to `/path/to/mvhuman/sequence/smpl_neutral`.
* Run the data processing code as below:
```
python preprocessing_mvhuman.py --cfg mvhuman/config/200173.yaml  # this is to get the data for Vid2Avatar and LS_Avatar
python preprocessing_humannerf.py --seq mvhuman/200173_training  # this is to get data for HumanNeRF and MonoHuman
python preprocessing_npc.py --seq mvhuman/200173_training  # this is to get data for NPC
```

## Process MonoPerfCap Data, YouTube Data and Custom Data
We follow [Vid2Avatar](https://github.com/MoyGcc/vid2avatar/tree/main) to process the custom data.
* We use [ROMP](https://github.com/Arthur151/ROMP#installation) to obtain initial SMPL shape and poses: 
```
pip install --upgrade simple-romp
```
* Download the model of [Segment Anything](https://github.com/facebookresearch/segment-anything) and move them to the corresponding place:
```
cd ../
mkdir model/
mv /path/to/sam_vit_h_4b8939.pth model/sam_vit_h_4b8939.pth
```
* Since the original [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md) is not easy to install, we integrate the pytorch implementation [pytorch_openpose](https://github.com/beingjoey/pytorch_openpose_body_25). Please follow [pytorch_openpose](https://github.com/beingjoey/pytorch_openpose_body_25) to download the pretrained model and put them into `openpose/model`:
```
cd openpose/model
mv /path/to/body_25.pth ./body_25.pth
mv /path/to/body_coco.pth ./body_coco.pth
mv /path/to/body_coco.pth ./body_coco.pth
```
* Put the video frames under the folder `../raw_data/{SEQUENCE_NAME}/frames`
* Modify the preprocessing script `run_preprocessing.sh` accordingly: the data source, sequence name, and the gender. The data source is by default "custom" which will estimate camera intrinsics. If the camera intrinsics are known, it's better if the true camera parameters can be given.
* Run preprocessing: 
```
bash run_preprocessing.sh
``` 
The processed data will be stored in `../data/`. The intermediate outputs of the preprocessing can be found at `../raw_data/{SEQUENCE_NAME}/`


## Acknowledgement
We have used codes from other great research work, including [Vid2Avatar](https://github.com/MoyGcc/vid2avatar), [pytorch_openpose](https://github.com/beingjoey/pytorch_openpose_body_25), [SMPL-X](https://github.com/vchoutas/smplx), [ActorHQ](https://github.com/synthesiaresearch/humanrf), [MvHumanNet](https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet). We sincerely thank the authors for their awesome work!
