from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import glob
from drawing_library import *


netMain = None
metaMain = None
altNames = None
IMAGE_PATH = None


def YOLO():
    global metaMain, netMain, altNames, IMAGE_PATH

    IMAGE_PATH = input("input path ")
    IMAGE_PATH += '/'
    OUTPUT_PATH = IMAGE_PATH + '/results/'
    try:
        os.makedirs(OUTPUT_PATH)
    except Exception as e:
        print(e)

    configPath = "/home/shamsher/weights/yolo/ANPR/4_class_new/t2.cfg"
    weightPath = "/home/shamsher/weights/yolo/ANPR/4_class_new/t2_b1.weights"
    metaPath = "/home/shamsher/weights/yolo/ANPR/4_class_new/t1.data"
    

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
    if not os.path.exists(IMAGE_PATH):
        raise ValueError("Invalid IMAGE_PATH `" +
                         os.path.abspath(IMAGE_PATH) + "`")
    if not os.path.exists(OUTPUT_PATH):
        raise ValueError("Invalid IMAGE_PATH `" +
                         os.path.abspath(OUTPUT_PATH) + "`")
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
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)

    for filename in glob.iglob(IMAGE_PATH + '*.jpg', recursive=True):
        print(filename)
        try:
            frame_read = cv2.imread(filename)
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5, nms=0.1)
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Demo', image)
            cv2.waitKey(0)
            cv2.imwrite(OUTPUT_PATH + '/' + os.path.basename(filename), image)
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    YOLO()
