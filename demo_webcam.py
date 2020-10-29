"""
https://github.com/cansik/yolo-hand-detection

Then run the following command to start a webcam detector with YOLOv3:
python demo_webcam.py

Or this one to run a webcam detrector with YOLOv3 tiny:
python demo_webcam.py -n tiny

For Yolov3-Tiny-PRN use the following command:
python demo_webcam.py -n prn
"""
import argparse
import cv2
import numpy as np

from yolo_hand import YOLO
from opencv_convex_hull import cv_process

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
vid_writer = cv2.VideoWriter('outputFile.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),round(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))))
crop_frame = None

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    width, height, inference_time, results = yolo.inference(frame)
    rects = []
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        rect = ((abs(y),abs(x)), (int((y+h)*1.2),int((x+w)*1.2)))
        rects.append(rect)
        #print(rect)

    if rects != []:
        for i, rect in enumerate(rects):
            cv_process(frame, rect, i)
    cv2.imshow("preview", frame)
    vid_writer.write(frame.astype(np.uint8))

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()