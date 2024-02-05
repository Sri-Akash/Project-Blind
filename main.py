import cv2

class ObjectDetector:
    def __init__(self, cascade_path):
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect_objects(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        objects = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return objects

def main():
    cascade_path = "haarcascade_frontalface_default.xml"  # Replace with the actual path
    object_detector = ObjectDetector(cascade_path)

    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        objects = object_detector.detect_objects(frame)

        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
