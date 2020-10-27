'''
Opencv, YOLO tutorial
video, webcom, image object detection - COCO
run module with command line arguments
    h: help
    i <image file>: run detection on image
    v <video file>: run detection on video
    none: run detection on webcam
https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
'''

import cv2 as cv
import os, sys, getopt
import numpy as np

def parse(argv):
   imagefile = ''
   videofile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:v:",["ifile=","vfile="])
   except getopt.GetoptError:
      print ('yolo_video.py -i <image file> -o <video file>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('yolo_video.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         imagefile = arg
      elif opt in ("-o", "--ofile"):
         videofile = arg
   return imagefile, videofile

image_file, video_file = parse(sys.argv[1:])


# Get the names of the output layers
# -----------------------------------
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
# --------------------------------
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

# Remove the bounding boxes with low confidence using non-maxima suppression
# ---------------------------------------------------------------------------
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

# 1 Initialize the parameters
# --------------------------
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
# inpWidth = 100       #Width of network's input image
# inpHeight = 100      #Height of network's input image

# 2 Load names of classes
# ----------------------
classesFile = "models/coco.names";
modelConfiguration = "models/yolov3.cfg";
modelWeights = "models/yolov3.weights";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Step 3 : Read the input
# In this step we read the image, video stream or the webcam. 
# In addition, we also open the video writer to save the frames 
# with detected output bounding boxes.
# -----------------------------
outputFile = "yolov3_video_out.avi"
if (image_file):
    # Open the image file
    if not os.path.isfile(image_file):
        print("Input image file ", image_file, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(image_file)
    outputFile = image_file[:-4]+'_yolov3_video_out.jpg'
elif (video_file):
    # Open the video file
    if not os.path.isfile(video_file):
        print("Input video file ", video_file, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(video_file)
    outputFile = video_file[:-4]+'_yolo_video_out.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)
    # Get the video writer initialized to save the output video
    # if (not image_file):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


# Step 4 : Process each frame
# The input image to a neural network needs to be in a certain format called a blob.
# -----------------------------------------------------------------------------------
while cv.waitKey(1) & 0xFF != ord('q'):
    
    # get frame from the video
    hasFrame, frame = cap.read()
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)
    
    # Put efficiency information. The function getPerfProfile returns the
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    # Write the frame with the detection boxes
    cv.imshow('frame', frame)
    if (image_file):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))



