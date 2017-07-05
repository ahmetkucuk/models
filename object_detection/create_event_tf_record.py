# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import tensorflow as tf
from utils import dataset_util
from PIL import Image
import hashlib
import io
import numpy as np
from sklearn.utils import shuffle
from cStringIO import StringIO

flags = tf.app.flags
flags.DEFINE_string('dataset_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('dataset_name', '', 'test or train')
FLAGS = flags.FLAGS

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242

ORIGINAL_IMAGE_SIZE = 2048


def read_event_records(path_to_records, dataset_type):

    images = []
    data = []
    labels = []
    label_txts = []
    bbox_map = {}
    label_map = {}
    label_text_map = {}

    ar_count = 0
    ch_count = 0
    sg_count = 0
    fl_count = 0

    with open(os.path.join(path_to_records, "event_records.txt"), "r") as f:

        for l in f.readlines():

            l = l.replace("\n", "")
            tuples = l.split("\t")

            start_time = tuples[2]
            start_year = start_time[:4]

            if dataset_type == "event_train":
                if start_year == "2015":
                    continue
            elif dataset_type == "event_test":
                if start_year != "2015":
                    continue

            label = 0
            label_txt = "none"
            if tuples[1] == "AR":
                label = 1
                label_txt = "ar"
                ar_count += 1
            elif tuples[1] == "CH":
                label = 2
                label_txt = "ch"
                ch_count += 1
            elif tuples[1] == "SG":
                label = 3
                label_txt = "sg"
                sg_count += 1
            elif tuples[1] == "FL":
                label = 4
                label_txt = "fl"
                fl_count += 1
            else:
                continue

            bbox = tuples[4]
            bbox = [float(i) for i in bbox.split("-")]

            width = abs(bbox[0] - bbox[2])
            height = abs(bbox[1] - bbox[3])

            if width < 16 or height < 16:
                continue

            if width > height:
                ratio = height / width
            else:
                ratio = width / height

            if ratio < 0.5:
                continue

            bbox = [(i/4096) for i in bbox]

            image_name = os.path.join(path_to_records, tuples[5])
            if not image_name in bbox_map.keys():
                bbox_map[image_name] = [bbox]
                label_map[image_name] = [label]
                label_text_map[image_name] = [label_txt]
            else:
                bbox_map[image_name].append(bbox)
                label_map[image_name].append(label)
                label_text_map[image_name].append(label_txt)

    print('AR COUNT: ' + str(ar_count))
    print('CH COUNT: ' + str(ch_count))
    print('SG COUNT: ' + str(sg_count))
    print('FL COUNT: ' + str(fl_count))

    for image in bbox_map.keys():
        images.append(image)
        data.append(bbox_map[image])
        labels.append(label_map[image])
        label_txts.append(label_text_map[image])
    return images, data, labels, label_txts


def _check_if_exist(filename):
    return tf.gfile.Exists(filename)


def _create_multi_wavelength_image(base_filename):

    file131 = base_filename + "_131.jpg"
    file171 = base_filename + "_171.jpg"
    file193 = base_filename + "_193.jpg"

    if not (_check_if_exist(file131) and _check_if_exist(file171) and _check_if_exist(file193)):
        return None

    img131 = Image.open(file131)
    img131.load()
    img131.thumbnail((2048, 2048), Image.ANTIALIAS)
    data131 = np.asarray(img131, dtype="uint8")

    img171 = Image.open(file171)
    img171.load()
    img171.thumbnail((2048, 2048), Image.ANTIALIAS)
    data171 = np.asarray(img171, dtype="uint8")

    img193 = Image.open(file193)
    img193.load()
    img193.thumbnail((2048, 2048), Image.ANTIALIAS)
    data193 = np.asarray(img193, dtype="uint8")

    rgbArray = np.zeros((ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE, 3), 'uint8')
    # rgbArray = np.zeros((ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE, 3), 'uint8')

    rgbArray[..., 0] = data131
    rgbArray[..., 1] = data171
    rgbArray[..., 2] = data193
    return rgbArray


def _process_image_and_create_example(filename, bboxes, labels, labels_txts):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    #filename = "/Users/ahmetkucuk/Documents/Research/solim_class/Bbox_Data/2012_01_01_04_00_00_171.jpg"

    image = _create_multi_wavelength_image(filename)

    if image is None:
        return

    image = Image.fromarray(image, "RGB")
    image.thumbnail((2048, 2048), Image.ANTIALIAS)
    image.save("temp.jpg")

    with tf.gfile.GFile("temp.jpg") as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

    if len(bboxes) != len(labels):
        raise ValueError("length of bboxes and labels are not same")

    # Read the XML annotation file.
    shape = [ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE, 3]
    # Find annotations.
    difficult_obj = []
    truncated = []
    poses = []

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for bbox in bboxes:
        xmin.append(float(bbox[0]))
        ymin.append(float(bbox[3]))
        xmax.append(float(bbox[2]))
        ymax.append(float(bbox[1]))

    for i in range(len(labels)):
        difficult_obj.append(0)
        truncated.append(0)
        poses.append('')

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(shape[0]),
        'image/width': dataset_util.int64_feature(shape[1]),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/key/sha256': dataset_util.bytes_feature(key),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(labels_txts),
        'image/object/class/label': dataset_util.int64_list_feature(labels),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))

    tf.gfile.Remove("temp.jpg")

    return example


def _add_to_tfrecord(name, bboxes, labels, labels_txts, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    example = _process_image_and_create_example(name, bboxes, labels, labels_txts)
    if example is not None:
        tfrecord_writer.write(example.SerializeToString())


def run(dataset_dir, output_dir, name='event_train', shuffling=True):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    images, data, labels, labels_txt = read_event_records(dataset_dir, dataset_type=name)

    if shuffling:
        images, data, labels, labels_txt = shuffle(images, data, labels, labels_txt, random_state=RANDOM_SEED)
    # Process dataset files.
    i = 0
    tf_filename = os.path.join(output_dir, ("%s.tfrecord" % name))
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        while i < len(images):
            sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(images)))
            sys.stdout.flush()
            image = images[i]
            _add_to_tfrecord(image, data[i], labels[i], labels_txt[i], tfrecord_writer)
            i += 1

            # Finally, write the labels file:
            # Open new TFRecord file.
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Event dataset!')


def main(_):
    dataset_name = FLAGS.dataset_name
    dataset_dir = FLAGS.dataset_dir
    output_dir = FLAGS.output_dir
    run(dataset_dir, output_dir, name=dataset_name)

'''
python create_event_tf_record.py \
    --dataset_name="event_test" \
    --dataset_dir="/Users/ahmetkucuk/Documents/Research/solim_class/Bbox_Data" \
    --output_dir="/Users/ahmetkucuk/Documents/Research/solim_class/tf_records_detection"

python create_event_tf_record.py \
    --dataset_name="event_train" \
    --dataset_dir="/home/ahmet/workspace/data/full_disk_171" \
    --output_dir="/home/ahmet/workspace/data/full_disk_171_detection_clean_dif0_tfrecords"


python create_event_tf_record.py \
    --dataset_name="event_train" \
    --dataset_dir="/home/ahmet/workspace/event-detection-data/full-disk-detection/data_new/" \
    --output_dir="/home/ahmet/workspace/event-detection-data/full-disk-detection/tfrecord_2048"
'''

if __name__ == '__main__':
    tf.app.run()