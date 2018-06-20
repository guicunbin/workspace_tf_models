#coding:utf-8
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time
import socket
from server_utils import *
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# This is needed since the notebook is stored in the object_detection folder.
TF_MODELS_DIR="/home/guicunbin/workspace/models/"
sys.path.append("{}/research/".format(TF_MODELS_DIR))
sys.path.append("{}/research/object_detection/".format(TF_MODELS_DIR))
from object_detection.utils import ops as utils_ops
# This is needed to display the images.
from utils import label_map_util
from utils import visualization_utils as vis_util
Session_config = tf.ConfigProto()
Session_config.gpu_options.allow_growth = True



sock = create_server_socket(TCP_IP=get_local_IP(), TCP_PORT=9999)

print "wait Source .... "
src, src_addr = sock.accept()
print "Source Connected by", src_addr


print "wait Destination ...."
dst, dst_addr = sock.accept()
print "Destination Connected by", dst_addr




PATH_TO_CKPT = '/home/guicunbin/workspace/models/research/my_workspace/models/faster_rcnn_inception_v2_coco/inference/frozen_inference_graph.pb'
#PATH_TO_CKPT = '/home/guicunbin/workspace/TF_ckpt/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = ('/home/guicunbin/workspace/models/research/object_detection/data/mscoco_label_map.pbtxt')
#PATH_TO_LABELS = ('/home/guicunbin/workspace/models/research/object_detection/data/ava_label_map_v2.1.pbtxt')
NUM_CLASSES = 90   #coco
#NUM_CLASSES = 545  #oid
#NUM_CLASSES = 80    #ava
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')



TEST_IMAGE_DIR  = "/home/guicunbin/workspace/dataset/pets/images"
TEST_IMAGE_PATHS= ["{}/{}".format(TEST_IMAGE_DIR,img_name) for img_name in os.listdir(TEST_IMAGE_DIR)]

image_H = 600; image_W = 600
with detection_graph.as_default():
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        if key + ':0'  in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(key + ":0")
    if 'detection_masks' in tensor_dict:
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


    sess = tf.Session(config = Session_config)
    #for image_path in TEST_IMAGE_PATHS:
    while(True):
        image = get_img_from_socket(src)
        image = cv2.resize(image, (image_H, image_W));
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        print "-->image.shape = {} ".format(image.shape)
        assert len(image.shape) == 3 and image.shape[0]==image_H and image.shape[1] == image_W;
        image_4d = np.expand_dims(image, 0)
        print "-->image_4d.shape = {}".format(image_4d.shape)
        feed_dict={image_tensor: image_4d}
        # Run inference
        t1 = time.time();
        #_ = sess.run(tensor_dict['detection_boxes'], feed_dict=feed_dict)
        output_dict = sess.run(tensor_dict,feed_dict = feed_dict)
        print "----->  using time = {} ms ".format(round((time.time() - t1)*1000, 0));

        #   #print _.shape
        #   output_dict = sess.run(tensor_dict,feed_dict = feed_dict)
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        #send_img_to_socket(dst, image)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        #cv2.imwrite("./dect_results.jpg", image)
        send_img_to_socket(dst, image)

sock.close();
dst.close();

