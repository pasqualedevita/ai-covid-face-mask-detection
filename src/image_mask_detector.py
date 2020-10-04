# USAGE
# python src/image_mask_detector.py --image testset/0001.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os


# load the face detector model from disk


def load_face_detector(modelDir):
    print("[INFO] loading face detector model...")
    for file in os.listdir(modelDir):
        if file.endswith(".prototxt"):
            prototxtPath = os.path.sep.join([modelDir, file])
        if file.endswith(".caffemodel"):
            weightsPath = os.path.sep.join([modelDir, file])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    return faceNet


# load the mask detector model from disk


def load_mask_detetcor(modelDir):
    print("[INFO] loading mask detector model...")
    for file in os.listdir(modelDir):
        if file.endswith(".model"):
            modelPath = os.path.sep.join([modelDir, file])
    maskNet = load_model(modelPath)
    return maskNet

# load the input image from disk


def load_image(imagePath):
    image = cv2.imread(imagePath)
    return image


# detect faces from images and extract ROIs


def detect_face(image, faceNet, confidence):
    print("[INFO] computing face detections...")

    # clone image, and grab the image spatial dimensions
    orig = image.copy()
    # get image height and width image.shape[:2]
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image,
                                 1.0,
                                 (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    ROIs = []

    # loop over the detections
    for detection in detections[0, 0]:
        # extract the confidence (i.e., probability) associated with the detection
        detectedConfidence = detection[2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if detectedConfidence > confidence:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detection[3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            # check if box dimension coherence
            if (startX >= endX) or (startY >= endY):
                continue

            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            ROIs.append([face, startX, endX, startY, endY])

    return ROIs


def detect_mask(maskNet, image, ROIs):
    print("[INFO] computing mask detections...")
    results = []

    for ROI in ROIs:
        face = ROI[0]
        startX = ROI[1]
        endX = ROI[2]
        startY = ROI[3]
        endY = ROI[4]

        (with_mask,
         without_mask,
         with_incorrect_mask) = maskNet.predict(face)[0]
        pred = "undefined"
        color = (0, 0, 0)

        max_value = max(with_mask,
                        without_mask,
                        with_incorrect_mask)

        if (max_value == with_incorrect_mask):
            pred = "with_incorrect_mask"
            # yellow
            color = (255, 255, 0)
        if (max_value == with_mask):
            pred = "with_mask"
            # green
            color = (0, 255, 0)
        if (max_value == without_mask):
            pred = "without_mask"
            # red
            color = (0, 0, 255)

        # include the probability in the label
        label = pred + ": " + "{:.2f}".format(max_value * 100)

        # display the label and bounding box rectangle on the output image
        cv2.putText(image,
                    label,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2)
        cv2.rectangle(image,
                      (startX, startY),
                      (endX, endY),
                      color,
                      2)

        results.append([pred, max_value, startX, endX, startY, endY])

    return (image, results)


def execute(image, face_detector='models/face_detector', mask_detector='models/mask_detector', confidence=0.6):
    faceNet = load_face_detector(face_detector)
    maskNet = load_mask_detetcor(mask_detector)
    image = load_image(image)
    ROIs = detect_face(image, faceNet, confidence)
    (image, results) = detect_mask(maskNet, image, ROIs)
    # show the output image
    # cv2.imshow("Output", image)
    # cv2.waitKey(5*1000)

    return (image, results)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",
                    "--image",
                    type=str,
                    default="testset/0009.jpeg",
                    help="path to input image")
    ap.add_argument("-f",
                    "--face_detector",
                    type=str,
                    default="models/face_detector",
                    help="path to face detector model")
    ap.add_argument("-m",
                    "--mask_detector",
                    type=str,
                    default="models/mask_detector",
                    help="path to mask detector model")
    ap.add_argument("-c",
                    "--confidence",
                    type=float,
                    default=0.6,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    execute(args["image"],
            args["face_detector"],
            args["mask_detector"],
            args["confidence"])
