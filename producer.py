import sys
import time
import cv2
from kafka import SimpleProducer, KafkaClient
from multiprocessing import Queue


import numpy as np
import argparse
import imutils
import time


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

IGNORE = set(["background", "aeroplane","bird", "boat","bottle", "bus","cat","cow", "diningtable","dog", "horse", "motorbike","pottedplant", "sheep", "train", "tvmonitor"])

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
proto = '/home/paperspace/project/MobileNetSSD_deploy.prototxt'
model = '/home/paperspace/project/MobileNetSSD_deploy.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto, model)



#  connect to Kafka
kafka = KafkaClient('localhost:9092')
producer = SimpleProducer(kafka)
# Assign a topic
topic = 'my-topic'


def video_emitter(video):
    val = 0
    # Open the video
    video = cv2.VideoCapture(video)
    video.set(cv2.CAP_PROP_FPS, 20)
    print video.get(cv2.CAP_PROP_FPS)
    vidQ = Queue()
    print(' emitting.....')
    while (video.isOpened):
        # read the image in each frame
        success, image = video.read()
        if not success:
            break
        if val != 2:
            val +=1
            continue

        val = 0

        (orgH,orgW) = image.shape[:2]
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()


        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] in IGNORE:
                    continue
                #drawing the pedictions on the frame!
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 1)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[idx], 2)

        image = imutils.resize(image, width=orgW, height=orgH)
        vidQ.put(image)
        while not vidQ.empty():
            # convert the image png
            ret, jpeg = cv2.imencode('.png', vidQ.get())
            # Convert the image to bytes and send to kafka
            producer.send_messages(topic, jpeg.tobytes())
            # To reduce CPU usage create sleep time of 0.2sec
            time.sleep(0.01)

    video.release()
    print('done emitting')

if __name__ == '__main__':
    #video_emitter('video.mp4')
    video_emitter("rtsp://localhost:2222/h264")
