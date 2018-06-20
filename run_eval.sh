#   source ./run_dependency.sh
TF_models_path="/home/guicunbin/workspace/models"
Date_suffix=$(date '+%Y_%m_%d_%H_%M_%S')
eval_log_file="./log/eval_log.log"

#   pipeline_config_path="/home/guicunbin/workspace/workspace_tf_models/model_configs/ssd_inception_v2_coco.config"
#   checkpoint_dir="/home/guicunbin/workspace/TF_ckpt/ssd_inception_v2_coco_2017_11_17/"
#   eval_dir="./log/eval/${Date_suffix}/"

pipeline_config_path="/home/guicunbin/workspace/workspace_tf_models/model_configs/ssd_inception_v2_coco.config"
checkpoint_dir="/home/guicunbin/workspace/workspace_tf_models/models/ssd_inception_v2_coco/train/"
eval_dir="./log/eval/${Date_suffix}/"



#   pipeline_config_path="/home/guicunbin/workspace/workspace_tf_models/model_configs/ssd_inception_v2_coco.config"
#   checkpoint_dir="/home/guicunbin/workspace/workspace_tf_models/models/ssd_inception_v2_coco/train/"
#   eval_dir="./log/eval/${Date_suffix}/"




#   pipeline_config_path="/home/guicunbin/workspace/workspace_tf_models/model_configs/faster_rcnn_inception_v2_coco.config"
#   checkpoint_dir="/home/guicunbin/workspace/TF_ckpt/faster_rcnn_inception_v2_coco_2018_01_28/"
#   eval_dir="./log/eval/${Date_suffix}/"



#   pipeline_config_path="/home/guicunbin/workspace/workspace_tf_models/model_configs/faster_rcnn_inception_v2_coco.config"
#   checkpoint_dir="/home/guicunbin/workspace/workspace_tf_models/models/faster_rcnn_inception_v2_coco/train/"
#   eval_dir="./log/eval/${Date_suffix}/"



#   pipeline_config_path="/home/guicunbin/workspace/workspace_tf_models/model_configs/faster_rcnn_inception_resnet_v2_atrous_DC_car.config"
#   #checkpoint_dir="/home/guicunbin/workspace/workspace_tf_models/models/faster_rcnn_resnet101_DC_car_train_all_vars/inference"
#   checkpoint_dir="/home/guicunbin/workspace/workspace_tf_models/models/faster_rcnn_inception_resnet_v2_atrous_DC_car/inference"
#   eval_dir="./log/eval/${Date_suffix}/"


cout_string="\n -->pipe:  ${pipeline_config_path}\n\n-->ckpt:  ${checkpoint_dir} \n\n -->eval:  ${eval_dir}"
echo -e "\n\n\n\n\n\n\n\n\n\n\n --- ${Date_suffix} ---\n" >> ${eval_log_file}
#echo -e $(cat ./run_dependency.sh) >> ${eval_log_file}
echo -e ${cout_string} | tee -a ${eval_log_file}


CUDA_VISIBLE_DEVICES=1 python ${TF_models_path}/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${pipeline_config_path} \
    --checkpoint_dir=${checkpoint_dir}\
    --eval_dir=${eval_dir} | tee -a ${eval_log_file}


python ./scripts/sendemail.py \
    --context "${Date_suffix}" \
    --subject "$0 finished ! "
