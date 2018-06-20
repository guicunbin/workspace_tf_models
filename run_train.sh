#source ./run_dependency.sh
TF_models_path="/home/guicunbin/workspace/models"



#   PATH_TO_YOUR_PIPELINE_CONFIG="/home/guicunbin/workspace/workspace_tf_models/model_configs/faster_rcnn_inception_v2_coco.config"
#   PATH_TO_TRAIN_DIR="/home/guicunbin/workspace/workspace_tf_models/models/faster_rcnn_inception_v2_coco/train"

PATH_TO_YOUR_PIPELINE_CONFIG="/home/guicunbin/workspace/workspace_tf_models/model_configs/ssd_inception_v2_coco.config"
PATH_TO_TRAIN_DIR="/home/guicunbin/workspace/workspace_tf_models/models/ssd_inception_v2_coco/train"


#   PATH_TO_YOUR_PIPELINE_CONFIG="/home/guicunbin/workspace/workspace_tf_models/model_configs/faster_rcnn_inception_resnet_v2_atrous_DC_car.config"
#   PATH_TO_TRAIN_DIR="/home/guicunbin/workspace/workspace_tf_models/models/faster_rcnn_inception_resnet_v2_atrous_DC_car/train/"



python ./scripts/remove_the_same_adjacent_row.py ${PATH_TO_YOUR_PIPELINE_CONFIG}
my_trainer_config=./scripts/my_trainer_config.py
gpu_fraction=0.5
save_interval_secs=3600
save_summaries_secs=3600

sed -i "s|per_process_gpu_memory_fraction.*|per_process_gpu_memory_fraction = ${gpu_fraction}|g" ${my_trainer_config}
sed -i "s|save_interval_secs.*|save_interval_secs = ${save_interval_secs}|g" ${my_trainer_config}
sed -i "s|save_summaries_secs.*|save_summaries_secs = ${save_summaries_secs}|g" ${my_trainer_config}






CUDA_VISIBLE_DEVICES=1 python ${TF_models_path}/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
