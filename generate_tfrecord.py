"""
Usage:

# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import sys
sys.path.append("../../models/research")

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#flags.DEFINE_string('label', '', 'Name of class label')
# if your image has more labels input them as
flags.DEFINE_string('label1', '', 'Name of class[1] label')
flags.DEFINE_string('label2', '', 'Name of class[2] label')
flags.DEFINE_string('label3', '', 'Name of class[3] label')
flags.DEFINE_string('label4', '', 'Name of class[4] label')
flags.DEFINE_string('label5', '', 'Name of class[5] label')
flags.DEFINE_string('label6', '', 'Name of class[6] label')
flags.DEFINE_string('label7', '', 'Name of class[7] label')
flags.DEFINE_string('label8', '', 'Name of class[8] label')
flags.DEFINE_string('label9', '', 'Name of class[9] label')
flags.DEFINE_string('label10', '', 'Name of class[10] label')
flags.DEFINE_string('label11', '', 'Name of class[11] label')
flags.DEFINE_string('label12', '', 'Name of class[12] label')

# and so on.
flags.DEFINE_string('img_path', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
# for multiple labels add more else if statements
def class_text_to_int(row_label):
    #if row_label == FLAGS.label:   'ship':
        #return 1
    # comment upper if statement and uncomment these statements for multiple labelling
     if row_label == FLAGS.label1:
       return 1
     elif row_label == FLAGS.label2:
       return 2
     elif row_label == FLAGS.label3:
       return 3
     elif row_label == FLAGS.label4:
       return 4
     elif row_label == FLAGS.label5:
       return 5
     elif row_label == FLAGS.label6:
       return 6
     elif row_label == FLAGS.label7:
       return 7
     elif row_label == FLAGS.label8:
       return 8
     elif row_label == FLAGS.label9:
       return 9
     elif row_label == FLAGS.label10:
       return 10
     elif row_label == FLAGS.label11:
       return 11
     elif row_label == FLAGS.label12:
       return 12
     else:
       return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        #classes_text.append(row['class'].encode('utf8'))
        classes_text.append(str(row['class']).encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
