import cv2
import pytesseract
import pyttsx3
from PIL import Image

# Tesseract configuration parameters
myconfig = r"--psm 6 --oem 3 -l eng"

# Pyttsx3 configuration parameters
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Change the index to select a different voice
engine.setProperty('rate', 150)  # Adjust the speaking rate
engine.setProperty('volume', 1.0)  # Adjust the speaking volume

video = cv2.VideoCapture(0)

captured_image = None

while True:
    success, frame = video.read()
    if success:
        cv2.imshow("Video", frame)

        key = cv2.waitKey(1)
        if key == 13:
            captured_image = frame.copy()
            cv2.imshow("Captured Image", captured_image)

        elif key == 27:
            break
    else:
        break

cv2.destroyAllWindows()

if captured_image is not None:
    # Convert the captured image to grayscale for better OCR results
    gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

    # Thresholding can be applied to enhance text visibility
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(Image.fromarray(thresholded), config=myconfig)
    print("Recognized Text:", text)

    # Enhance the voice by adjusting pyttsx3 parameters
    engine.say(text)
    engine.runAndWait()