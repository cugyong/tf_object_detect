# encoding:utf-8
import tensorflow as tf
import numpy as np

import os
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

# 下载下来的模型的目录
MODEL_DIR = 'my_images/export_dir/'
# 下载下来的模型的文件
MODEL_CHECK_FILE = os.path.join(MODEL_DIR, 'frozen_inference_graph.pb')
# 数据集对于的label
MODEL_LABEL_MAP = os.path.join('data', 'unionpay.pbtxt')
# 数据集分类数量，可以打开pascal_label_map.pbtxt文件看看
MODEL_NUM_CLASSES = 2

# 这里是获取实例图片文件名，将其放到数组中
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGES_PATHS = []
for i in range(1, 4):
    TEST_IMAGES_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'unionpay_'+str(i)+'.jpg'))
print(TEST_IMAGES_PATHS)

# TEST_IMAGES_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'two_logo.jpg')]

# 输出图像大小，单位是in
IMAGE_SIZE = (12, 8)

tf.reset_default_graph()

# 将模型读取到默认的图中
with tf.gfile.GFile(MODEL_CHECK_FILE, 'rb') as fd:
    _graph = tf.GraphDef()
    _graph.ParseFromString(fd.read())
    tf.import_graph_def(_graph, name='')

# 加载pascal数据标签
label_map = label_map_util.load_labelmap(MODEL_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=MODEL_NUM_CLASSES)
category_index = label_map_util.create_category_index(categories)


# 将图片转化成numpy数组形式
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# 在图中开始计算
detection_graph = tf.get_default_graph()
with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGES_PATHS:
        print(image_path)
        # 读取图片
        image = Image.open(image_path)
        # 将图片数据转成数组
        image_np = load_image_into_numpy_array(image)
        # 增加一个维度
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # 下面都是获取模型中的变量，直接使用就好了
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # 存放所有检测框
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # 每个检测结果的可信度
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # 每个框对应的类别
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # 检测框的个数
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # 开始计算
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                            feed_dict={image_tensor: image_np_expanded})
        # 打印识别结果
        print(num_detections)
        print(boxes)
        print(classes)
        print(scores.max)

        # 得到可视化结果
        vis_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8
        )
        # 显示
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.show()
