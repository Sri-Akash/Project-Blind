import cv2
import numpy as np
import pyttsx3
from playsound import playsound  # for playing sound

# Constants and colors
Known_distance = 76.2  # cm
Known_width = 14.3  # cm
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
fonts = cv2.FONT_HERSHEY_COMPLEX

# Load car cascade classifier
car_cascade = cv2.CascadeClassifier("cars.xml")


# Focal length finder function (unchanged)
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


# Distance estimation function
def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    distance = (real_object_width * Focal_Length) / object_width_in_frame
    return distance


# Object detection and distance estimation function
def car_detection_and_distance(image):
    height, width, _ = image.shape

    # Convert to grayscale for car detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect cars
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Loop through detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)

        # Calculate distance for detected car
        object_width = w
        distance = Distance_finder(Focal_length_found, Known_width, object_width)

        cv2.line(image, (30, 30), (230, 30), RED, 32)
        cv2.line(image, (30, 30), (230, 30), BLACK, 28)

        cv2.putText(image, f"Car: {round(distance, 2)} CM", (30, 35), fonts, 0.6, GREEN, 2)

        # Play beep sound for cars within 200 cm
        if distance <= 200:
            playsound("Images/beep.mp3")  # Replace "beep.wav" with your sound file

    return image


# Read reference image and calculate focal length (unchanged)
ref_image = cv2.imread("ReferenceImages/image1.png")
ref_image_face_width = 0  # assuming no face detection needed for focal length calculation
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_face_width)

# Initialize camera
cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()

    # Call car detection and distance function
    frame = car_detection_and_distance(frame)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
