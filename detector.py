import cv2
import numpy as np
import pyttsx3

Known_distance = 76.2
Known_width = 14.3
GREEN = (0, 255, 0)


class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()
        self.tts = pyttsx3.init()

        # Initialize ref_image_face_width as a class variable
        self.ref_image_face_width = 0

    def readClasses(self):
        with open(self.classesPath, 'rt') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')

        self.colorList = np.random.uniform(low=8, high=255, size=(len(self.classesList), 3))

    def Focal_Length_Finder(self, measured_distance, real_width, width_in_rf_image):
        # finding the focal length
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length

    def Distance_finder(self, Focal_Length, real_face_width, face_width_in_frame):
        distance = (real_face_width * Focal_Length) / face_width_in_frame

        # return the distance
        return distance

    def speak_text(self, text):
        # Set properties (optional)
        self.tts.setProperty('rate', 150)  # Speed percent (can go over 100)
        self.tts.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        # Speak the given text
        self.tts.say(text)
        # Wait for the speech to finish
        self.tts.runAndWait()

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening file ...")
            return

        success, image = cap.read()

        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)

            if classLabelIDs is not None and len(self.classesList) > 0:
                bboxs = list(bboxs)
                confidences = list(np.array(confidences).reshape(1, -1)[0])
                confidences = list(map(float, confidences))

                bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

                if len(bboxIdx) != 0:
                    for i in range(0, len(bboxIdx)):
                        bbox = bboxs[np.squeeze(bboxIdx[i])]
                        classConfidence = confidences[np.squeeze(bboxIdx[i])]
                        classLabelID = int(np.squeeze(classLabelIDs[int(np.squeeze(bboxIdx[i]))]))

                        if 0 <= classLabelID < len(self.classesList):
                            classLabel = self.classesList[classLabelID]
                            classColor = [int(c) for c in self.colorList[classLabelID]]

                            displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                            x, y, w, h = bbox
                            self.ref_image_face_width = w if w > 0 else self.ref_image_face_width
                            Focal_Length = self.Focal_Length_Finder(Known_distance, Known_width, w)
                            Distance = self.Distance_finder(Focal_Length, Known_distance, self.ref_image_face_width)

                            cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                            cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                            cv2.putText(
                                image, f"Distance: {round(Distance, 2)} CM", (30, 35),
                                cv2.FONT_HERSHEY_PLAIN, 0.6, GREEN, 2)

                            # Speak the distance
                            self.speak_text(f"{classLabel} {round(Distance, 2)} centimeters away from you")

            cv2.imshow("Result", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            success, image = cap.read()
        cv2.destroyAllWindows()
        cap.release()


