from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet


from ctypes import *
import math
import random
import os
import cv2
import time
from drawing_library import *

netMain = None
metaMain = None
altNames = None


def YOLO():
    global metaMain, netMain, altNames
    
    frame_id = 1
    VIDEO_PATH = input("input path ")
    OUTPUT_PATH = './results/'
    try:
        os.makedirs(OUTPUT_PATH)
    except Exception as e:
        print(e)

    configPath = "/home/shamsher/weights/yolo/ANPR/4_class_new/t1.cfg"
    weightPath = "/home/shamsher/weights/yolo/ANPR/4_class_new/t1_b1.weights"
    metaPath = "/home/shamsher/weights/yolo/ANPR/4_class_new/t1.data"

    configPath = "./yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"

    #configPath = "/home/shamsher/weights/yolo/ANPR/4_class/ees_24092020_435000_v1.cfg"
    #weightPath = "/home/shamsher/weights/yolo/ANPR/4_class/ees_24092020_435000_v1.weights"
    #metaPath = "/home/shamsher/weights/yolo/ANPR/4_class/coco.data"
    
    
    
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(VIDEO_PATH)
    #cap.set(3, 1280)
    #cap.set(4, 720)
    out = cv2.VideoWriter(
        OUTPUT_PATH + "/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain),darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    #darknet_image = darknet.make_image(416, 416, 3)
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    while True:
        try:
            ret, frame_read = cap.read()
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.3, nms=0.3)
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.putText(image, str(frame_id), (10,10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1)
            cv2.imshow('Demo', image)
            cv2.waitKey(3)
            out.write(image)
            print(frame_id)
            frame_id += 1
        except Exception as e:
            print(e)
            break
    cap.release()
    out.release()


if __name__ == "__main__":
    YOLO()

