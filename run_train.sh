source ./run_dependency.sh
CUDA_VISIBLE_DEVICES=1 python ${TF_models_path}/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
