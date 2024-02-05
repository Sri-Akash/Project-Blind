import cv2
import numpy as np

# Distance from camera to object measured in centimeters
Known_distance = 76.2

# Width of the object in the real world or Object Plane (centimeters)
Known_width = 14.3

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")


# Function to find focal length
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


# Function to find distance
def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    distance = (real_object_width * Focal_Length) / object_width_in_frame
    return distance


# Function to find object width in frame
def object_data(frame):
    object_width = 0
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect objects using MobileNet SSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Class ID 15 corresponds to "person" in MobileNet SSD
                object_width = int(detections[0, 0, i, 3] * frame.shape[1])
                break
    return object_width


# Reading reference image from directory
ref_image = cv2.imread("Images/hotel.jpg")
# Find the object width(pixels) in the reference image
ref_image_object_width = object_data(ref_image)

# Get the focal length by calling "Focal_Length_Finder"
Focal_length_found = Focal_Length_Finder(
    Known_distance, Known_width, ref_image_object_width
)

# Show the reference image
cv2.imshow("ref_image", ref_image)

# Initialize the camera object
cap = cv2.VideoCapture(0)

while True:
    # Reading the frame from the camera
    _, frame = cap.read()

    # Calling object_data function to find the width of the object(pixels) in the frame
    object_width_in_frame = object_data(frame)

    # Check if the object is detected
    if object_width_in_frame != 0:
        # Finding the distance by calling function
        Distance = Distance_finder(
            Focal_length_found, Known_width, object_width_in_frame
        )

        # Draw line as background of text
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)

        # Drawing Text on the screen
        cv2.putText(
            frame,
            f"Distance: {round(Distance, 2)} CM",
            (30, 35),
            fonts,
            0.6,
            GREEN,
            2,
        )

    # Show the frame on the screen
    cv2.imshow("frame", frame)

    # Quit the program if you press 'q' on the keyboard
    if cv2.waitKey(1) == ord("q"):
        break

# Closing the camera
cap.release()

# Closing the windows that are opened
cv2.destroyAllWindows()
