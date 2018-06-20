#source ./run_dependency.sh
#---> find the latest_num for ckpt



TF_models_path="/home/guicunbin/workspace/models"
this_model_root_path="/home/guicunbin/workspace/workspace_tf_models/models/faster_rcnn_inception_resnet_v2_atrous_DC_car"
PATH_TO_TRAIN_DIR="${this_model_root_path}/train"
EXPORT_DIR="${this_model_root_path}/inference"
PATH_TO_YOUR_PIPELINE_CONFIG="${PATH_TO_TRAIN_DIR}/pipeline.config"
mkdir ${EXPORT_DIR}





for fi in $(ls -t $(find ${PATH_TO_TRAIN_DIR} -name "*model.ckpt*"));
do 
    latest_num=$(echo $fi | awk -F'.meta' '{print $1}' | awk -F'.ckpt-' '{print $2}'); 
    break; 
done;

trained_checkpoint_prefix=${PATH_TO_TRAIN_DIR}/model.ckpt-${latest_num}
echo_string="\n\n\n\n\n\n   ${PATH_TO_YOUR_PIPELINE_CONFIG}\n  ${trained_checkpoint_prefix}\n ${EXPORT_DIR}\n"




echo -e "${echo_string}"



CUDA_VISIBLE_DEVICES=1 python ${TF_models_path}/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${trained_checkpoint_prefix} \
    --output_directory ${EXPORT_DIR} \
&&  echo -e "${echo_string}" >> ${EXPORT_DIR}/export.log
