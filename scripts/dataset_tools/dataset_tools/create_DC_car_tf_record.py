import hashlib
import io
import json
import os
import numpy as np
import PIL.Image
import sys
import cv2
TF_models_path="/home/guicunbin/workspace/models/"
sys.path.append(TF_models_path)
sys.path.append("{}/research/".format(TF_models_path))

from pycocotools import mask
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
tf.flags.DEFINE_boolean('include_masks', False,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_dir', '',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '',
                       'Validation image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

tf.flags.DEFINE_integer('car_cate_index', 3, 'car_cate_index')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)





def create_tf_example(image_id,
                      filename,
                      annotations_list,
                      image_dir,
                      category_of_eachbox,
                      include_masks=False):
    image_full_path = "{}/{}".format(image_dir, filename);
    image_np = cv2.imread(image_full_path);

    image_height, image_width, image_channel = image_np.shape;

    with tf.gfile.GFile(image_full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for idx_of_box in range(len(annotations_list)):
      x, y, width, height = annotations_list[idx_of_box]
      if width <= 0 or height <= 0:
        num_annotations_skipped += 1
        continue
      if x + width > image_width or y + height > image_height:
        num_annotations_skipped += 1
        continue
      xmin.append(float(x) / image_width)
      xmax.append(float(x + width) / image_width)
      ymin.append(float(y) / image_height)
      ymax.append(float(y + height) / image_height)
      is_crowd.append(0);
      category_id = category_of_eachbox[idx_of_box];
      category_ids.append(category_id)
      category_names.append("car".encode('utf8'))
      area.append(10.00)

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped






def _create_tf_record_from_coco_annotations(annotations_file, image_dir, output_path, include_masks, car_cate_index = 3):
    print "\n\n\n ====>  car_cate_index = {}\n\n\n".format(car_cate_index);
    images = open(annotations_file).readlines();
    missing_annotation_count = 0
    total_num_annotations_skipped = 0
    tf.logging.info('writing to output path: %s', output_path)
    writer = tf.python_io.TFRecordWriter(output_path)

    for idx, one_line in enumerate(images[1:]):
        if idx % 100 == 0:  
            tf.logging.info('On image %d of %d', idx, len(images))
        one_line = one_line.strip();
        one_line_li = one_line.split(',');
        if(len(one_line_li[1])==0): 
            missing_annotation_count += 1;
            continue;

        image_name = one_line_li[0]
        annotations_list =  one_line_li[1].split(';');
        if(len(annotations_list[-1])==0):
            annotations_list.pop();
        category_of_eachbox = [car_cate_index for i in range(len(annotations_list))]
        for i in range(len(annotations_list)):
            try:
                annotations_list[i] = [int(float(num_str)) for num_str in  annotations_list[i].split('_')]
            except:
                print "{}/ line: {}".format(i, idx);
                print annotations_list
                sys.exit(0);

        _, tf_example, num_annotations_skipped = create_tf_example(idx,
            image_name, annotations_list, image_dir, category_of_eachbox, include_masks)
        writer.write(tf_example.SerializeToString())
        total_num_annotations_skipped += num_annotations_skipped;
    writer.close()
    tf.logging.info('%d images are missing annotations.',missing_annotation_count)
    tf.logging.info('Finished writing, skipped %d annotations.',total_num_annotations_skipped)





def main(_):
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = "{}/DC_car_cate_index_{}_train_.record".format(FLAGS.output_dir,  FLAGS.car_cate_index)
  val_output_path = "{}/DC_car_cate_index_{}_val.record".format(FLAGS.output_dir,       FLAGS.car_cate_index)


  _create_tf_record_from_coco_annotations(
      FLAGS.train_annotations_file,
      FLAGS.train_image_dir,
      train_output_path,
      FLAGS.include_masks,
      FLAGS.car_cate_index)
  _create_tf_record_from_coco_annotations(
      FLAGS.val_annotations_file,
      FLAGS.val_image_dir,
      val_output_path,
      FLAGS.include_masks,
      FLAGS.car_cate_index)





if __name__ == '__main__':
  tf.app.run()
