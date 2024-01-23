from detector import *
import os


def main():
    videoPath = 0
    configPath = os.path.join('ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    modelPath = os.path.join('frozen_inference_graph.pb')
    classesPath = os.path.join('labels.txt')

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()


if __name__ == '__main__':
    main()
