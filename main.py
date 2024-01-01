import cv2

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For Image

'''image = cv2.imread('Images/2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = dataset.detectMultiScale(gray)

for x, y, w, h in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (235, 81, 23), 2)

cv2.imshow('Single', image)'''

# For WebCam

video = cv2.VideoCapture(0)
while True:
    success, frame = video.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (235, 81, 23), 2)
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            break
    else:
        break
