from ultralytics import YOLO
import cv2
import cvzone
import requests
from pprint import pprint
import math
from sort import *

from datetime import datetime, timedelta
# from gpiozero import LED

# ledRed = LED(18)
# ledYellow = LED(2)
# ledGreen = LED(3)

# ledRed = LED(16)
# ledYellow = LED(20)
# ledGreen = LED(21)
now = datetime.now()

# ------new----------

red = False
yellow = False
green = True


# def onOffFunc():
#     global red, yellow, green, now
#     while True:
#         nowNew = datetime.now()
#         # print(nowNew,"\n", (now + timedelta(seconds=5 )),"\n",nowNew == (now + timedelta(seconds=5 )))
#
#         if green == True and nowNew.strftime("%H:%M:%S") == (now + timedelta(seconds=10)).strftime("%H:%M:%S"):
#             print("times matched green", green)
#             green = False
#             yellow = True
#             now = datetime.now()
#
#         elif yellow == True and nowNew.strftime("%H:%M:%S") == (now + timedelta(seconds=10)).strftime("%H:%M:%S"):
#             print("This is yellow")
#             yellow = False
#             red = True
#             now = datetime.now()
#         elif red == True and nowNew.strftime("%H:%M:%S") == (now + timedelta(seconds=20)).strftime("%H:%M:%S"):
#             print("This is red")
#             red = False
#             green = True
#             now = datetime.now()
#             now = datetime.now()
#         if red == True:
#             ledRed.on()
#         else:
#             ledRed.off()
#
#         if yellow == True:
#             ledYellow.on()
#         else:
#             ledYellow.off()
#
#         if green == True:
#             ledGreen.on()
#         else:
#             ledGreen.off()
#
#         return red


# import RPI.GPIO as GPIO


# cap = cv2.VideoCapture("../Videos/cars.mp4")
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

regions = ['lk']  # Change to your country
count = 0

# inputPin = 18


# cap.set(3,1280)
# cap.set(4,720)
# cap.set(3,416)
# cap.set(4,416)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH,416);
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,416);


width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)
model = YOLO("Yolo-Weights/yolov8n.pt")

# classNames = ["person", "car", "motorbike", "bus", "truck","cell phone"
#               ]

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("images/MaskProduction1.png")
# mask = cv2.imread("images/webcam2.png")


tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [0, 325, 680, 325]
totalCount = []

yx1 = 0
yx2 = 0
yy2 = 0
yy1 = 0
yw = 0
yh = 0

# GPIO.setmode(inputPin,GPIO.IN,pull_up_down=GPIO.PUD_UP)

savedImg = ""

while True:
    # while onOffFunc():
    while True:
        # if GPIO.input(inputPin):
        #     pass
        # else:
        success, img = cap.read()
        imgRegion = cv2.bitwise_and(img, mask)
        # imgRegion = img

        results = model(imgRegion, stream=True)
        # results = model(img,stream=True)
        # results = model(img,stream=True)
        detections = np.empty((0, 5))

        if cv2.waitKey(1) & 0xFF == ord("s"):
            cv2.imwrite("./plates/scanned_img_" + str(count) + ".jpg", imgRegion)
            cv2.rectangle(img, (0, 200), (640, 300), cv2.FILLED)
            # cv2.putText(img,"Plate Saved",(150,265),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),2)
            cv2.imshow("Results", img)
            # cv2.waitKey(500)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # print(x1,y1,x2,y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

                # x1,y1,w,h = box.xywh[0];
                w, h = x2 - x1, y2 - y1

                # bbox = int(x1),int(y1),int(w),int(h)

                cvzone.cornerRect(img, (x1, y1, w, h), l=9)

                conf = math.ceil((box.conf[0] * 100)) / 100
                # print(conf)

                cls = int(box.cls[0])
                try:
                    currentClass = classNames[cls]
                    if classNames[cls] == "car" or classNames[cls] == "bus" or classNames[cls] == "truck" or classNames[
                        cls] == "motorbike":
                        currentClass = classNames[cls]
                    elif classNames[cls + 1] == "car" or classNames[cls + 1] == "bus" or classNames[
                        cls + 1] == "truck" or classNames[cls + 1] == "motorbike":
                        currentClass = classNames[cls + 1]
                    elif classNames[cls + 2] == "car" or classNames[cls + 2] == "bus" or classNames[
                        cls + 2] == "truck" or classNames[cls + 2] == "motorbike":
                        currentClass = classNames[cls + 2]
                    elif classNames[cls + 3] == "car" or classNames[cls + 3] == "bus" or classNames[
                        cls + 3] == "truck" or classNames[cls + 3] == "motorbike":
                        currentClass = classNames[cls + 3]
                    print(classNames[cls + 1], "///////////////////////")
                    print(classNames[cls + 2], "///////////////////////")
                    print(classNames[cls + 3], "///////////////////////")
                except:

                    print("there was a error")
                    currentClass = "none"
                    print(currentClass)
                # if currentClass != "none":
                #     print(currentClass,"--------------------------")

                # if len(currentClass) > 0:
                # if currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike"  and conf > 0.3:
                if currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike" and conf > 0.2:
                    # -----------New Code ------------------------
                    print(currentClass, "---------------------")
                    cv2.imwrite("./plates/scanned_img_" + str(count) + ".jpg", img)
                    savedImg = "./plates/scanned_img_" + str(count) + ".jpg"
                    # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
                    # cv2.rectangle(img,(0,200),(640,300),cv2.FILLED)
                    # cv2.putText(img,"Plate Saved",(150,265),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),2)
                    # cv2.imshow("Results", img)
                    # cv2.waitKey(5000)
                    # ----------------NEW Code---------------------
                    # cvzone.putTextRect(img,f"{classNames[cls]}-{conf}",(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    yx1 = x1
                    yx2 = x2
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f"{id}", (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2

            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, y2), 5, (255, 0, 255), cv2.FILLED)
            cv2.line(img, ((x1 + w) // 2, y1), ((x1 + w) // 2, h), (0, 0, 255), 5)

            # cy= cy*2

            # if limits[0] < y1 < limits[2] and limits[1] -15 < h < limits[1] + 15:
            # if limits[0] < cx < limits[2] and limits[1] - 15 < y2 < limits[1] + 15 or limits[1] - 15 < y1 < limits[1] + 15:
            print(limits[1] - 100, cy, y1, y2, limits[1] + 100)
            if limits[0] < cx < limits[2] and limits[1] - 50 < (cy + 50) < limits[1] + 50:

                print("passed checked", totalCount, id)
                print(totalCount.count(id))
                # if limits[0] < x1 < limits[2] or  limits[0] < x2 < limits[2] and limits[1] - 15 < y1 < limits[1] + 15:
                if totalCount.count(id) == 0:
                    print("passed checked insideas", totalCount, id, count)
                    totalCount.append(id)
                    # cv2.waitKey(50)

                    # if cv2.waitKey(1) & 0xFF == ord("s"):
                    #     -------------SCREENSHOT CODE --------------------
                    #     cv2.imwrite("./plates/scanned_img_" + str(count) + ".jpg", img)
                    #     # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
                    #     # cv2.rectangle(img,(0,200),(640,300),cv2.FILLED)
                    #     cv2.putText(img,"Plate Saved",(150,265),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),2)
                    #     cv2.imshow("Results", img)
                    #     cv2.waitKey(5000)
                    # -------------SCREENSHOT CODE --------------------

                    print(count)

                    # with open("./plates/scanned_img_" + str(count) + ".jpg", 'rb') as fp:
                    with open(savedImg, 'rb') as fp:
                        response = requests.post(
                            'https://api.platerecognizer.com/v1/plate-reader/',
                            data=dict(regions=regions),  # Optional
                            files=dict(upload=fp),
                            headers={'Authorization': 'Token 2c3a6cd14c3343ce0cd16c3a7c74895aceab0c4a'})
                    pprint(response.json())

                    if (len(response.json()["results"])):
                        print(response.json()["results"][0]["plate"])

                        response = requests.get(
                            f'https://iot-backend-vosb.onrender.com/api/v1/cases/handleVehicleNumber/{response.json()["results"][0]["plate"]}/Homagama')
                        # data=dict(regions=regions),  # Optional
                        # files=dict(upload=fp),
                        # headers={'Authorization': 'Token 2c3a6cd14c3343ce0cd16c3a7c74895aceab0c4a'})

                        # cv2.waitKey(500)

                        if response:
                            print(response)

                        # Calling with a custom engine configuration
                        # import json
                        #
                        # with open("./plates/scanned_img_" + str(count) + ".jpg", 'rb') as fp:
                        #     response = requests.post(
                        #         'https://api.platerecognizer.com/v1/plate-reader/',
                        #         data=dict(regions=['us-ca'], config=json.dumps(dict(region="strict"))),  # Optional
                        #         files=dict(upload=fp),
                        #         headers={'Authorization': '2c3a6cd14c3343ce0cd16c3a7c74895aceab0c4a'})
                        #     print(totalCount.count(id))
                        cv2.waitKey(500)
                    count += 1

        cvzone.putTextRect(img, f"{len(totalCount)}", (50, 50))

        # cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

        cv2.imshow("Image", img)
        cv2.imshow("ImageRegion", imgRegion)
        cv2.waitKey(1)
