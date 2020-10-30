import cv2
import numpy as np
import math
import time


def cv_process(img, capture_rect, from_main):
    
    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, capture_rect[0], capture_rect[1], (0, 255, 0), 5)
    crop_img = img[capture_rect[0][0]:capture_rect[1][0], capture_rect[0][1]:capture_rect[1][1]]
    #cv2.imshow("cropped", crop_img)

    # convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh1 = cv2.bitwise_not(thresh1)

    # show thresholded image
    #cv2.imshow('Thresholded', thresh1)

    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')
    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '4':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (255, 255, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # calculate areas for cnt, hull and ratio
    hull_area = cv2.contourArea(hull)
    cnt_area = cv2.contourArea(cnt)
    ratio_area = (hull_area - cnt_area) / cnt_area

    # drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    if from_main:
        result_drawing = np.zeros(crop_img.shape,np.uint8)
    else:
        result_drawing = np.zeros((300,300),np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # # applying Cosine Rule to find angle for all defects (between fingers)
    # # with angle > 90 degrees and ignore defects
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            #dist = cv2.pointPolygonTest(cnt,far,True)

            # draw a line from start to end i.e. the convex points (finger tips)
            # (can skip this part)
            cv2.line(crop_img,start, end, [0, 255, 0], 2)
            #cv2.circle(crop_img,far, 5, [0, 0, 255], -1)
    # define actions required

    # show appropriate images in windows
    #cv2.imshow('Gesture', img)
    #now = datetime.datetime.now().second
    seconds = int((time.perf_counter() % 3) + 1)
    finger = finger_count(count_defects)
    text = stein_schere_papier(finger, hull_area, cnt_area, ratio_area)
    cv2.putText(result_drawing, "Time: "+str(seconds), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    cv2.putText(result_drawing, "Hull: "+str(hull_area), (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    cv2.putText(result_drawing, "CNT: "+str(cnt_area), (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    cv2.putText(result_drawing, "Ratio: "+str(ratio_area), (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    cv2.putText(result_drawing, "Finger: "+str(finger), (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    cv2.putText(result_drawing, text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    if from_main:
        all_img = np.hstack((drawing, result_drawing))
        cv2.imshow('Contours', all_img)
    else:
        cv2.imshow('draw', drawing)
        cv2.imshow('result', result_drawing)

def finger_count(count_defects):
    if count_defects == 1:
        finger_count = 2
    elif count_defects == 2:
        finger_count = 3
    elif count_defects == 3:
        finger_count = 4
    elif count_defects == 4:
        finger_count = 5
    else:
        finger_count = 0
    return finger_count

def stein_schere_papier(finger, hull, cnt, ratio):
    if (finger == 2 or finger == 3) and ratio > 0.2 and hull < 40000:
        text = "Schere"
    elif ratio < 0.2 and hull < 30000:
        text = "Stein"
    else:
        text = "Papier"
    return text

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    capture_rect = ((100, 100), (400, 400))

    while(cap.isOpened()):
        # read image
        ret, img = cap.read()
        cv_process(img, capture_rect, True)
        k = cv2.waitKey(10)
        if k == 27:
            break