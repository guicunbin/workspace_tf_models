source ./run_dependency.sh
cat ./run_dependency.sh


#checkpoint_dir="/home/guicunbin/workspace/TF_ckpt/faster_rcnn_nas_coco_2018_01_28/"
checkpoint_dir=${PATH_TO_TRAIN_DIR}



echo "-->pipe:  ${PATH_TO_YOUR_PIPELINE_CONFIG}"
echo "-->ckpt:  ${checkpoint_dir}"
echo "-->eval:  ${PATH_TO_EVAL_DIR}"
CUDA_VISIBLE_DEVICES=1 python ../object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${checkpoint_dir}\
    --eval_dir=${PATH_TO_EVAL_DIR}
