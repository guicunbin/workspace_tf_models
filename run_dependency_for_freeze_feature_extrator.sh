my_workspace="/home/guicunbin/workspace/workspace_tf_models"
TF_models_path="/home/guicunbin/workspace/models"
models_configs_path="/home/guicunbin/workspace/models/research/object_detection/samples/configs"


#model_name="faster_rcnn_resnet101_DC_car_train_all_vars"
model_name="faster_rcnn_resnet101_DC_car"
fine_tune_checkpoint="/home/guicunbin/workspace/TF_ckpt/faster_rcnn_resnet101_kitti_2018_01_28/model.ckpt"
from_detection_checkpoint="true"
train_tfrecord_path="/home/guicunbin/workspace/dataset/DC_car_detect/DC_car_cate_index_1_train_.record"
val_tfrecord_path="/home/guicunbin/workspace/dataset/DC_car_detect/DC_car_cate_index_1_val.record"
label_map_path="/home/guicunbin/workspace/dataset/DC_car_detect/DC_car_label_map.pbtxt"
metrics_set="coco_detection_metrics"
freeze_variables=".*FeatureExtractor.*"








freeze_variables_string="freeze_variables: \"${freeze_variables}\""
metrics_set_string="metrics_set: \"${metrics_set}\""
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




sed -i "s|from_detection_checkpoint:.*|from_detection_checkpoint: ${from_detection_checkpoint}|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s|fine_tune_checkpoint_type:.*|from_detection_checkpoint: ${from_detection_checkpoint}|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}



sed -i "s|\".*label_map.pbtxt\"|\"${label_map_path}\"|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s|\".*train.*record\"|\"${train_tfrecord_path}\"|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s|\".*val.*record\"|\"${val_tfrecord_path}\"|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s|fine_tune_checkpoint: \".*\"|fine_tune_checkpoint: \"${fine_tune_checkpoint}\"|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}



sed -i "/${metrics_set_string}/d" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "/${freeze_variables_string}/d" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s| num_steps:| #num_steps:|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "/eval_config:.*/a\  ${metrics_set_string}"  ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "/train_config:.*/a\ ${freeze_variables_string}"  ${PATH_TO_YOUR_PIPELINE_CONFIG}
