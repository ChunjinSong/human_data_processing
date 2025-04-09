# pre-define data
source="custom"
seq="sequence_training"
gender="NEUTRAL"


# run ROMP to get initial SMPL parameters
echo "Running ROMP"
romp --mode=video --calc_smpl --render_mesh -i=./raw_data/$seq/frames -o=./raw_data/$seq/ROMP --smpl_path ./model/romp/SMPL_$gender.pth

# obtain the projected masks through estimated perspective camera (so that OpenPose detection)
echo "Getting projected SMPL masks"
python preprocessing.py --source $source --seq $seq --gender $gender --mode mask

# run OpenPose to get 2D keypoints
echo "Running OpenPose"
python run_openpose.py --seq $seq

# offline refine poses
echo "Refining poses offline"
python preprocessing.py --source $source --seq $seq --gender $gender --mode refine

## scale images and center the human in 3D space
echo "Scaling images and centering human in 3D space"
python preprocessing.py --source $source --seq $seq --gender $gender --mode final --scale_factor 1

## normalize cameras such that all cameras are within the sphere of radius 3
echo "Normalizing cameras"
python normalize_cameras.py --input_cameras_file ./data/$seq/cameras.npz \
                            --output_cameras_file ./data/$seq/cameras_normalize.npz


echo "Running SAM"
python run_sam.py --seq $seq

echo "Processing for lsavatar"
python preprocessing_lsavatar.py --seq $seq --gender $gender

echo "Processing humannerf/monohuman"
python preprocessing_humannerf.py --seq $seq --gender $gender

echo "Processing pmavatar/npc data"
python preprocessing_pmavatar.py --seq $seq --gender $gender

echo $seq "finished"



