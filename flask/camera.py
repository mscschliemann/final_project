import cv2
from yolo_hand import YOLO

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        self.yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def get_boxed_frame(self):
        rval, frame = self.video.read()
        width, height, inference_time, results = self.yolo.inference(frame)
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()