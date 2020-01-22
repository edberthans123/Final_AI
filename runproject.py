import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

global kamera
kamera = cv2.VideoCapture(0)

def video():

    while kamera.isOpened():

        status, frame = kamera.read()

        if not status:
            break

        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')

        print(bbox, label, conf)

        out = draw_bbox(frame, bbox, label, conf, write_conf=True)

        cv2.imshow("Real-time object detection", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kamera.release()
    cv2.destroyAllWindows()

def live():
    loadvideo = cv2.VideoCapture('data/demo.mp4')


    if not loadvideo.isOpened():
        print("Could not open video")
        exit()

    while loadvideo.isOpened():
        status, frame = loadvideo.read()

        if not status:
            break

        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')

        print(bbox, label, conf)

        out = draw_bbox(frame, bbox, label, conf, write_conf=True)

        cv2.imshow("Real-time object detection", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    loadvideo.release()
    cv2.destroyAllWindows()

print(50*"*")
print("\t\t OBJECT RECOGNISE.....")
print(50*"*")

choice = ["Video Processing", "Live Processing"]
run = True
for x in range(len(choice)):
    print(x+1, " : ", choice[x])
print(50*"*")

while run:
    select = int(input("Select Task : "))
    if select == 1:
        live()
    elif select == 2:
       video()
    else:
        print(50*"*")
        print("\t\t Unavailable Task.....")
        print(50*"*")


