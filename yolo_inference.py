# USAGE
# python video_yolo.py

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from imutils.video import VideoStream

class MyCam:

    def __init__(self):
        self.vs = VideoStream(src=1).start()

    def get_image(self):
        return self.vs.read()

class Yolo:

    def __init__(self):

        yolo_path = 'yolo/yolo_coco'

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([yolo_path, "coco.names"])
        # labelsPath = "coco.names"
        self.LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
            dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
        configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def infer(self, frame):

        # if the frame dimensions are empty, grab them
        # if W is None or H is None:
        H, W = frame.shape[:2]

        print('size inside yolo', frame.shape)

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        objects = []

        confidence_threshold = 0.25  # set the threshold for confidence
        overlap_threshold = 0.3  # set the threshold for NMS overlap

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > confidence_threshold:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,
            overlap_threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                x1 = x
                y1 = y
                x2 = int(x + w)
                y2 = int(y + h)

                # xc = int(x + w/2)
                # yc = int(y + h/2)

                objects.append([self.LABELS[classIDs[i]], x1, y1, x2, y2])

                # # draw a bounding box rectangle and label on the frame
                # color = [int(c) for c in self.COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],
                #     confidences[i])
                # cv2.putText(frame, text, (x, y - 5),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # show the output frame
        # cv2.imshow("Output", frame)

        return frame, objects


if __name__ == '__main__':

    my_cam = MyCam()  # create a cam object
    yolo = Yolo()  # create a Yolo object

    # run in forever loop
    while True:

        frame = my_cam.get_image()  # get the frame

        yolo.infer(frame)  # run inference on frame

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break














