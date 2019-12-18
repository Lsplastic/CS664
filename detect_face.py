#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import Tkinter 
import tkFileDialog


from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './output_model/frozen_inference_graph.pb'
# PATH_TO_CKPT = './output_model/1/saved_model.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#print(label_map)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#print(categories)
category_index = label_map_util.create_category_index(categories)
#print(category_index)

class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=self.detection_graph, config=config) as self.sess:

                self.windowNotSet = True


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        print(len(image_np_expanded))
        print(len(image_np_expanded[0]))
        print(len(image_np_expanded[0][0]))
        print(len(image_np_expanded[0][0][0]))

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)

class FaceDetection(object):
    def __init__(self):
        self.filename = ''
        self.root = Tkinter.Tk()
        self.root.title("FACE DETECTION")
        self.video_button = Tkinter.Button(self.root,width = 20,text = "CAMERA MODE",command = detect_face)
        self.label_1 = Tkinter.Label(self.root, text="------------------------------------------",width = 30)
        self.label_2 = Tkinter.Label(self.root, text='IMAGE INPUT MODE',bg="green",width=30)
        self.label = Tkinter.Label(self.root, text='Please choose the input image',width=30)
        self.photo_input = Tkinter.Label(self.root,text = '',width = 30)
        self.photo_select = Tkinter.Button(self.root,width = 20, text = "Select File", command = self.select_file)
        self.detect_button = Tkinter.Button(self.root,width = 20, text = "IMAGE MODE",command = self.get_photo_name)

    def gui_arrang(self):
        self.video_button.pack()
        self.label_1.pack()
        self.label_2.pack()
        self.label.pack()
        self.photo_input.pack()
        self.photo_select.pack()
        self.detect_button.pack()

    def get_photo_name(self):   
        # photo_name = self.photo_input.get()
        # image = cv2.imread('%s'%photo_name)
        # if(isinstance(image,None)):
        photo_name = self.filename
        flag = cv2.imread('%s'%photo_name)
        if flag is None:
            self.photo_input.config(text='Please choose the input image!')
        else:
            detect_photo(self.filename)

    def select_file(self):
        self.filename = ''
        self.filename =tkFileDialog.askopenfilename()
        self.photo_input.config(text=self.filename)

def detect_face():
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    # cap = cv2.VideoCapture(camID)
    cap = cv2.VideoCapture(0)
    windowNotSet = True
    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        print(image.shape)
        [h, w] = image.shape[:2]
        print("#============================#")
        print h, w
        print("#============================#")
        image = cv2.flip(image, 1)

        (boxes, scores, classes, num_detections) = tDetector.run(image)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        if windowNotSet is True:
            cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            windowNotSet = False

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
    cv2.destroyAllWindows() 

def detect_photo(photo_name):
    file_name = photo_name
    print(file_name)
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    # cap = cv2.VideoCapture(camID)
    #cap = cv2.VideoCapture(0)
    windowNotSet = True
    image = cv2.imread('%s'%file_name)

    # if ret == 0:
    #     print("error")
    [h, w] = image.shape[:2]
    print h, w
    image = cv2.flip(image, 1)

    (boxes, scores, classes, num_detections) = tDetector.run(image)
    # print(np.squeeze(boxes))
    # print(np.squeeze(scores))
    # print(num_detections)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)

    if windowNotSet is True:
        cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
        windowNotSet = False

    cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q') or k == 27:
        cv2.destroyAllWindows() 
    

    


if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 2:
    #     print """usage:%s (cameraID | filename)
    #             Dectect faces in the video
    #             example:
    #             %s 0
    #             """ % (sys.argv[0], sys.argv[0])
    #     exit(1)

    # try:
    #     camID = int(sys.argv[1])
    # except:
    #     camID = sys.argv[1]
    #----------------------------------------------#
    # tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    # # cap = cv2.VideoCapture(camID)
    # cap = cv2.VideoCapture(0)
    # windowNotSet = True
    # while True:
    #     ret, image = cap.read()
    #     if ret == 0:
    #         break

    #     [h, w] = image.shape[:2]
    #     print h, w
    #     image = cv2.flip(image, 1)

    #     (boxes, scores, classes, num_detections) = tDetector.run(image)

    #     vis_util.visualize_boxes_and_labels_on_image_array(
    #         image,
    #         np.squeeze(boxes),
    #         np.squeeze(classes).astype(np.int32),
    #         np.squeeze(scores),
    #         category_index,
    #         use_normalized_coordinates=True,
    #         line_thickness=4)

    #     if windowNotSet is True:
    #         cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
    #         windowNotSet = False

    #     cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
    #     k = cv2.waitKey(1) & 0xff
    #     if k == ord('q') or k == 27:
    #         break

    # cap.release()
    # 初始化对象
    FD = FaceDetection()
    # 进行布局
    FD.gui_arrang()
    # 主程序执行
    Tkinter.mainloop()
    pass


