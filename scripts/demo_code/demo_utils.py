#coding: utf-8
import cv2
import time
import numpy as np
import tensorflow as tf
import sys
TF_MODELS_DIR="/home/guicunbin/workspace/models/"
sys.path.append("{}/research/".format(TF_MODELS_DIR))
sys.path.append("{}/research/object_detection/".format(TF_MODELS_DIR))
from object_detection.utils import ops as utils_ops



def get_now_time():
    return time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))



#TODO delete the boxes overlap > 90% 
def DC_car_dect_write_one_line_to_submit(f_write, image_np, image_name, image_H, image_W, output_dict, dect_cate_list = [3,4,5,6,7,8,9],  dect_score_min = 0.5, show_box_on_image=False):
    ###----------------------------------------------------------------------
    one_line = image_name + ",";
    print image_name; 
    score_min = 0.3
    cnt_i = 0;
    for i in range(output_dict['num_detections']):
        #   print "{}/{} scores : {}".format(i,output_dict['num_detections'],output_dict["detection_scores"][i])
        if(output_dict['detection_scores'][i] < dect_score_min): break;
        if(output_dict['detection_classes'][i] not in dect_cate_list): continue;

        one_line += "" if cnt_i==0 else ";";
        cnt_i += 1;

        y_l, x_l, y_r, x_r =   output_dict['detection_boxes'][i];
        x = int(image_W * x_l);
        y = int(image_H * y_l);
        w = int(image_W * (x_r - x_l));
        h = int(image_H * (y_r - y_l));
        x_y_w_h = "{}_{}_{}_{}".format(x,y,w,h);
        one_line += x_y_w_h;
        if show_box_on_image:
            #   print "--->  xyxy = ",x,y,x+w,y+h
            cv2.rectangle(image_np, (x,y), (x+w, y+h), (0, 0, 255), thickness = 4, lineType = 1 ,shift =0)
    one_line += "\n";
    if not show_box_on_image:
        f_write.write(one_line);



def restore_graph_from_pbfile(graph_path):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def get_num_classes_from_labeltxt(PATH_TO_LABELS):
    lines = open(PATH_TO_LABELS, "r").readlines()[::-1];
    for line in lines:
        if("id:" in line):
            num = int(line.strip().split(':')[-1].strip())
            return num;
    raise Exception('   not get the NUM_CLASSES !!!!')


def expand_this_graph_by_process_detection_masks(tensor_dict, image_H, image_W):
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image_H, image_W)
    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
    

def convert_output_dict_to_suitable_dtype(output_dict):
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    

def save_params_to_log(log_file, *args, **kwargs):
    save_var_dict = locals();
    log_string = "\n\n\n\n===>==>  time is {}  <==\n".format(get_now_time());
    for key in save_var_dict:
        log_string += "\n{}={}\n".format(key, save_var_dict[key]);
    for key in kwargs:
        log_string += "\n{}={}\n".format(key, kwargs[key]);
    with open(log_file, 'a') as fa:
        fa.write(log_string);
