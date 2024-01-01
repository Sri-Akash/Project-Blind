import cv2
import pytesseract
import pyttsx3
from PIL import Image

myconfig = r"--psm 6 --oem 3"

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
    gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(Image.fromarray(gray), config=myconfig)
    print(text)

    tts = pyttsx3.init()
    tts.say(text)
    tts.runAndWait()
