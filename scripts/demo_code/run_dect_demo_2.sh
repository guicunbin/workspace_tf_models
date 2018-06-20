#CUDA_VISIBLE_DEVICES=1 python ./dect_demo_by_src_and_dst.py
#CUDA_VISIBLE_DEVICES=1 python ./dect_demo_by_IpCamera_and_dst.py


Date_suffix=$(date '+%Y_%m_%d_%H_%M_%S')
dataset_root="/home/guicunbin/workspace/dataset/DC_car_detect/"
submit_file="${dataset_root}/submit/submit_${Date_suffix}.csv"
#submit_file="${dataset_root}/submit/submit_2018_06_04_15_29_23.csv"
log_file="${dataset_root}/run_log.txt"
graph_path="/home/guicunbin/workspace/workspace_tf_models/models/faster_rcnn_inception_resnet_v2_atrous_DC_car/inference/frozen_inference_graph.pb"
graph_path_log="/home/guicunbin/workspace/workspace_tf_models/models/faster_rcnn_inception_resnet_v2_atrous_DC_car/inference/export.log"
label_path='/home/guicunbin/workspace/workspace_tf_models/label_pbtxt/mscoco_label_map.pbtxt'
#label_path='/home/guicunbin/workspace/workspace_tf_models/label_pbtxt/kitti_label_map.pbtxt'
test_image_dir="/home/guicunbin/workspace/dataset/DC_car_detect/test_a/"
#test_image_dir="/home/guicunbin/workspace/dataset/DC_car_detect/train/"
DC_dect_cate_list="3,5,6,7,8,9,62" # for mscoco
#DC_dect_cate_list="1,"
DC_dect_score_min="0.8"
is_socked="1"
TCP_PORT=9999


echo -e "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n------------------->${Date_suffix}<--------------- \n" >> ${log_file}
echo -e $(cat ${graph_path_log}) >> ${log_file}

CUDA_VISIBLE_DEVICES=1 python ./dect_demo.py \
    --submit_file ${submit_file} \
    --log_file ${log_file} \
    --graph_path ${graph_path} \
    --label_path ${label_path} \
    --test_image_dir ${test_image_dir} \
    --DC_dect_cate_list ${DC_dect_cate_list} \
    --DC_dect_score_min ${DC_dect_score_min} \
    --is_socked ${is_socked} \
    --TCP_PORT ${TCP_PORT}

python ../sendemail.py \
    --context "${Date_suffix}" \
    --subject "$0 finished "
