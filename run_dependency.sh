my_workspace="/home/guicunbin/workspace/workspace_tf_models"
TF_models_path="/home/guicunbin/workspace/models"
models_configs_path="/home/guicunbin/workspace/models/research/object_detection/samples/configs"

#   #model_name="faster_rcnn_nas_coco"
#   #model_name="ssd_inception_v2_coco"
#   #model_name="ssd_mobilenet_v2_coco"
#   model_name="ssd_mobilenet_v1_coco"
#   fine_tune_checkpoint="/home/guicunbin/workspace/TF_ckpt/mobile_net_v1/mobilenet_v1_1.0_224.ckpt"
#   # "COCO//mscoco*"  can't be recognition
#   PATH_TO_BE_CONFIGURED="/home/guicunbin/workspace/dataset/COCO"


model_name="faster_rcnn_inception_resnet_v2_atrous_DC_car"
fine_tune_checkpoint="/home/guicunbin/workspace/TF_ckpt/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/model.ckpt"
# "COCO//mscoco*"  can't be recognition
PATH_TO_BE_CONFIGURED="/home/guicunbin/workspace/dataset/DC_car_detect"
dataset_name="DC_car"




my_workspace_model_path="${my_workspace}/models/${model_name}"
PATH_TO_YOUR_PIPELINE_CONFIG="${my_workspace}/model_configs/${model_name}.config"
PATH_TO_TRAIN_DIR="${my_workspace_model_path}/train"
PATH_TO_EVAL_DIR="${my_workspace_model_path}/eval"
EXPORT_DIR="${my_workspace_model_path}/inference"
mkdir  ${my_workspace_model_path}
mkdir  ${PATH_TO_TRAIN_DIR}
mkdir  ${PATH_TO_EVAL_DIR}
mkdir  ${EXPORT_DIR}


#cp  ${models_configs_path}/${model_name}.config  ${PATH_TO_YOUR_PIPELINE_CONFIG}




#   sed -i "s|from_detection_checkpoint: true|from_detection_checkpoint: false|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
#   
#   sed -i 's|fine_tune_checkpoint_type:  "detection"|from_detection_checkpoint: false|g' ${PATH_TO_YOUR_PIPELINE_CONFIG}



sed -i "s|mscoco|${dataset_name}|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s|"PATH_TO_BE_CONFIGURED/model.ckpt"|"${fine_tune_checkpoint}"|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s|PATH_TO_BE_CONFIGURED|"${PATH_TO_BE_CONFIGURED}"|g"  ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s| num_steps:| #num_steps:|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i '/eval_config:.*/a\  metrics_set: "coco_detection_metrics"'  ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i '/train_config:.*/a\ freeze_variables: ".*FeatureExtractor.*"'  ${PATH_TO_YOUR_PIPELINE_CONFIG}
