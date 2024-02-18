import cv2
import numpy as np
import pyttsx3
import threading

# Constants and colors
Known_distance = 76.2
Known_width = 14.3
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
fonts = cv2.FONT_HERSHEY_COMPLEX
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


# Distance estimation function
def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    distance = (real_object_width * Focal_Length) / object_width_in_frame
    return distance


# Object detection function
def object_detection(image):
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Keep only the detection with the highest confidence
    if boxes:
        max_confidence_index = np.argmax(confidences)
        x, y, w, h = boxes[max_confidence_index]

        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)

        object_width = w
        distance = Distance_finder(Focal_length_found, Known_width, object_width)

        cv2.line(image, (30, 30), (230, 30), RED, 32)
        cv2.line(image, (30, 30), (230, 30), BLACK, 28)

        cv2.putText(image, f"{classes[class_ids[max_confidence_index]]} {round(distance, 2)} CM", (30, 35),
                    fonts, 0.6, GREEN, 2)

        # Speak the distance in a separate thread
        threading.Thread(target=speak_distance, args=(classes[class_ids[max_confidence_index]], distance)).start()

    return image

# Function to speak the distance


def speak_distance(object_class, distance):
    engine = pyttsx3.init()
    engine.say(f"{object_class} is {round(distance, 2)} centimeters away from you.")
    engine.runAndWait()
    engine.stop()


# Function for face detection
def face_data(image):
    face_width = 0

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)
        face_width = w

    return face_width


# Read reference image
ref_image = cv2.imread("ReferenceImages/image1.png")
ref_image_face_width = face_data(ref_image)
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_face_width)

# Load YOLOv4-tiny model
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    face_width_in_frame = face_data(frame)
    if face_width_in_frame != 0:
        Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)

        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)

        cv2.putText(frame, f"Distance: {round(Distance, 2)} CM", (30, 35), fonts, 0.6, GREEN, 2)

    # Call object detection function
    frame = object_detection(frame)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
