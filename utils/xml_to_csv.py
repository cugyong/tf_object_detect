# coding:utf-8
# import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('image_path', '', 'Path to the image annotations xml input')
flags.DEFINE_string('csv_output', '', 'Path to the CSV output')
FLAGS = flags.FLAGS


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # image_path = os.path.join(os.getcwd(), 'annotations')  # os.getcwd()方法用于返回当前工作目录
    xml_df = xml_to_csv(FLAGS.image_path)
    # xml_df.to_csv('raccoon_labels.csv', index=None)
    xml_df.to_csv(FLAGS.csv_output, index=None)
    print('Successfully converted xml to csv.')


main()
