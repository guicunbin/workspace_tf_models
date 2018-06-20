my_workspace="/home/guicunbin/workspace/workspace_tf_models"
TF_models_path="/home/guicunbin/workspace/models"
models_configs_path="/home/guicunbin/workspace/models/research/object_detection/samples/configs"


model_name="faster_rcnn_inception_v2_coco"
fine_tune_checkpoint="/home/guicunbin/workspace/TF_ckpt/inception_v2.ckpt"
from_detection_checkpoint="false"
train_tfrecord_path="/home/guicunbin/workspace/dataset/COCO/mscoco_train.record"
val_tfrecord_path="/home/guicunbin/workspace/dataset/COCO/mscoco_val.record"
label_map_path="/home/guicunbin/workspace/dataset/COCO/mscoco_label_map.pbtxt"
metrics_set="coco_detection_metrics"








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





### ===========>    change the ${PATH_TO_YOUR_PIPELINE_CONFIG}


#cp  ${models_configs_path}/${model_name}.config  ${PATH_TO_YOUR_PIPELINE_CONFIG}


sed -i "s|from_detection_checkpoint:.*|from_detection_checkpoint: ${from_detection_checkpoint}|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s|fine_tune_checkpoint_type:.*|from_detection_checkpoint: ${from_detection_checkpoint}|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}


sed -i "s|\".*label_map.pbtxt\"|\"${label_map_path}\"|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s|\".*train.*record\"|\"${train_tfrecord_path}\"|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "s|\".*val.*record\"|\"${val_tfrecord_path}\"|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}

sed -i "s| num_steps:| #num_steps:|g" ${PATH_TO_YOUR_PIPELINE_CONFIG}
sed -i "/eval_config:.*/a\  ${metrics_set_string}"  ${PATH_TO_YOUR_PIPELINE_CONFIG}

python ./scripts/remove_the_same_adjacent_row.py ${PATH_TO_YOUR_PIPELINE_CONFIG}


