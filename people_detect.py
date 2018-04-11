# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
from datetime import datetime, timezone
import cv2
import sys
import time
import pytz
from matplotlib import pyplot as plt
from scipy.spatial import distance
import uuid

# Function to get Distance between two points (Euclidean Distance)
def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

# Function to inform door status.... actually only when it opens
def sendDoorStatus():
    dataToSend = {
        "Description": "Door Opened",
        "Source": "door_opened"
    }
    sendStatus(dataToSend)

# Function to inform Person Detection
def sendPeople(description):
    dataToSend = {
        "Description": description,
        "Source": "people_detect"
    }
    sendStatus(dataToSend)

# Function to actually do the HTTP Post
def sendStatus(data):

    fmt = "%Y-%m-%d %H:%M:%S.%f"
    date = datetime.now(timezone.utc)

    data["TimeStamp"] = date.strftime(fmt)
    data["EVENT_ID"] = str(uuid.uuid4())

    import requests
    from json import dumps
    API_ENDPOINT = 'http://localhost:3002/events'
    r = requests.post(url = API_ENDPOINT, data = dumps(data))

    # extracting response text
    pastebin_url = r.text
    print("The pastebin URL is:%s"%pastebin_url)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

# Set some camera parameters
cam_params = {
    'brightness': camera.get(11), #32.0,
    'contrast': camera.get(12), #64.0,
    'saturation': camera.get(13), #0.0,
    'hue' : camera.get(14), #,
    'gain': camera.get(15), #,
    'exposure': camera.get(16), #,
    }

cv2.namedWindow('After', cv2.WINDOW_NORMAL)

# These will store the place where the user last clicked
ix,iy = -1,-1
# mouse callback function
def getMousePoint(event,x,y,flags,param):
    global ix,iy, selectedPoint
    if event == cv2.EVENT_LBUTTONDBLCLK:
        ix,iy = x,y
        danger_area_pts1[selectedPoint - 1, 0] = x
        danger_area_pts1[selectedPoint - 1, 1] = y
        selectedPoint += 1
        if selectedPoint > len(danger_area_pts1):
            selectedPoint = 0

        print(danger_area_pts1)

selectedPoint = 1

# let the hardcoding begin...
# define danger / safe areas ... will use later on
danger_area_pts1 = np.array([[89,2],[198,3],[177,246],[79,247]], np.int32)
danger_area_pts2 = np.array([[381,1],[497,1],[436,248],[336,248]], np.int32)
exit_area_pts = np.array([[203,1],[380,1],[336,248],[183,248]], np.int32)

# define where we do the person detection
exit_area_detection = (180,00,371,154)

# our "Marker" for detecting open door... I use the handle (see video)
coord_door_open = (175, 118, 178, 122)
status_door = False

cv2.setMouseCallback('After',getMousePoint)
numPeople = 0
frame_counter = 0
lastFound = np.array((0,0,0))
door_status_sent = False

# this is part of the hack for not sending multiple HTTP Posts for the same
# person
personSentAt = time.time()

while(True):
    ret, frame = camera.read()

    # little code for looping on the video
    frame_counter += 1
    if frame_counter == camera.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0 #Or whatever as long as it is the same as next line
        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)


    image = imutils.resize(frame, width=500)

    # begin Door Open Detection
    door_handle_gray = image[coord_door_open[1]:coord_door_open[3], coord_door_open[0]:coord_door_open[2]]
    avg = door_handle_gray.mean()
    if avg < 100:
        status_door = True
        if not door_status_sent:
            sendDoorStatus()
            door_status_sent = True
    else:
        status_door = False
        door_status_sent = False

    # end Door Open Detection


    # begin People Detection
    image_for_detection = image[exit_area_detection[1]:exit_area_detection[3], exit_area_detection[0]:exit_area_detection[2]]
    (rects, weights) = hog.detectMultiScale(image_for_detection, winStride=(2, 2),
    padding=(2, 2), scale=1.05)

    rects = np.array([[x + exit_area_detection[0],
                       y + exit_area_detection[1],
                       x  + exit_area_detection[0] + w,
                       y + exit_area_detection[1] + h] for (x, y, w, h) in rects])

    pick = non_max_suppression(rects, probs=None, overlapThresh=0.05)
    # end People Detection

    cnts = np.array([exit_area_pts])

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (250, 255, 100), 2)
        cv2.circle(image, (xA+int((xB-xA)/2) , yA+int((yB-yA)/2)), 10, (250, 250, 250), -1 )


        # Check to see if the person was at the indicated zone
        coef = int((yB-yA) * 0.2)
        ret = cv2.pointPolygonTest(cnts, (xA + int((xB-xA) / 2), yB-coef), False)
        if ret > 0:
            # Check to see is not too soon to send a new HTTP Post (If it is,
            # it means is the same person)
            if (time.time() - personSentAt) > 2:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 0, 255), 2)
                lastFound = np.array((xA, yA,0))
                numPeople += 1
                print("Person Detected ({})".format(numPeople))

                personSentAt = time.time()
                sendPeople("Person detected on video")

        cv2.rectangle(image, (xA, yA), (xB, yB), (100, 255, 100), 1)

    # Draw Danger / Safe zones
    cv2.polylines(image, np.array([danger_area_pts1]), True, (0,0,255), 2)
    cv2.polylines(image, np.array([danger_area_pts2]), True, (0,0,255), 2)

    if status_door:
        cv2.polylines(image, np.array([exit_area_pts]), True, (0,255,0), 2)
    else:
        cv2.polylines(image, np.array([exit_area_pts]), True, (0,0,255), 2)

    cv2.imshow("After", image)

    k = cv2.waitKey(1)
    if k==27:
        break
    elif k==ord('q'):
        break
    elif k == ord('d'):
        if debug:
            debug = False
            printCamStats = False
        else:
            debug = True
            printCamStats = True
    elif k == ord('1'):
        selectedPoint = 1
    elif k == ord('2'):
        selectedPoint = 2
    elif k == ord('3'):
        selectedPoint = 3
    elif k == ord('4'):
        selectedPoint = 4
    elif k == ord('5'):
        selectedPoint = 5
    elif k == ord('6'):
        selectedPoint = 6
    elif k == ord('7'):
        selectedPoint = 7

camera.release()
cv2.destroyAllWindows()
