from detector import *
import os


def main():
    videoPath = 'Images/street.mp4'
    configPath = os.path.join('ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    modelPath = os.path.join('frozen_inference_graph.pb')
    classesPath = os.path.join('labels.txt')

    Detector(videoPath, configPath, modelPath, classesPath)


if __name__ == '__main__':
    main()