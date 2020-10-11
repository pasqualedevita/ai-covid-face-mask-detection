# USAGE
# python src/eval_mask_detector.py

import argparse
import csv
import os
import cv2
import image_mask_detector  # Importing specific function


def clean_dir(path):
    if os.path.exists(path):
        os.system("rm -rf "+path)
    os.makedirs(path)


def execute_mask_detector(testset_data, output, face_detector, mask_detector, confidence):
    clean_dir(output)
    # clean_dir(output+"with_incorrect_mask/")
    # clean_dir(output+"with_mask/")
    # clean_dir(output+"without_mask/")

    output_results = []
    for data in testset_data:
        image_path = data[0]
        label = data[1]
        image_mask_detector_result = image_mask_detector.execute(
                image_path,
                face_detector,
                mask_detector,
                confidence)
        if not os.path.exists(output+label):
            os.makedirs(output+label)
        cv2.imwrite(output+label+"/"+os.path.basename(image_path),image_mask_detector_result[0])
        output_results.append([label,image_mask_detector_result])
    
    return output_results


def eval_accuracy(mask_detector_results, output):
    
    with_mask_detected = 0
    without_mask_detected = 0
    with_incorrect_mask_detected = 0

    with_mask_expected = 0
    without_mask_expected = 0
    with_incorrect_mask_expected = 0

    for mask_detector_result in mask_detector_results:
        
        label_expected = mask_detector_result[0]
        print("label_expected: "+label_expected)

        if (label_expected == 'with_mask'):
            with_mask_expected = with_mask_expected + 1
        if (label_expected == 'without_mask'):
            without_mask_expected = without_mask_expected + 1
        if (label_expected == 'with_incorrect_mask'):
            with_incorrect_mask_expected = with_incorrect_mask_expected + 1
        
        if (len(mask_detector_result[1][1]) == 1):
            label_detected = mask_detector_result[1][1][0][0]
            print("label_detected: "+label_detected)
            if (label_detected == label_expected):
                if (label_detected == 'with_mask'):
                    with_mask_detected = with_mask_detected + 1
                if (label_detected == 'without_mask'):
                    without_mask_detected = without_mask_detected + 1
                if (label_detected == 'with_incorrect_mask'):
                    with_incorrect_mask_detected = with_incorrect_mask_detected + 1

    accuracy_with_mask = with_mask_detected / with_mask_expected
    accuracy_without_mask = without_mask_detected / without_mask_expected
    accuracy_with_incorrect_mask = with_incorrect_mask_detected / with_incorrect_mask_expected

    accuracy = (with_mask_detected + without_mask_detected + with_incorrect_mask_detected) / (with_mask_expected + without_mask_expected + with_incorrect_mask_expected)

    with open(output+'accuracy.txt', 'w') as output:
        output.write("accuracy_with_mask: "+str(accuracy_with_mask))
        output.write("\n")
        output.write("accuracy_without_mask: "+str(accuracy_without_mask))
        output.write("\n")
        output.write("accuracy_with_incorrect_mask: "+str(accuracy_with_incorrect_mask))
        output.write("\n")
        output.write("accuracy: "+str(accuracy))
    return accuracy


def load_testset_data(testset='testset/'):
    imagePaths = []
    testset_data = []
    valid_images = [".jpg", ".jpeg", ".png"]
    for dirname, dirs, filenames in os.walk(testset, topdown=True):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext.lower() not in valid_images:
                continue
            imagePaths.append(os.path.join(dirname, filename))

    # loop over the image paths
    for imagePath in imagePaths:
    # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        testset_data.append([imagePath,label])
    
    return testset_data


def execute(testset='testset/', output='output/', face_detector='models/face_detector/', mask_detector='models/mask_detector/', confidence=0.6):
    
    testset_data = load_testset_data(testset)
    
    mask_detector_results = execute_mask_detector(testset_data,
                                                  output,
                                                  face_detector,
                                                  mask_detector,
                                                  confidence)

    eval_accuracy(mask_detector_results, output)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",
                    "--testset",
                    type=str,
                    default="testset/",
                    help="path to input testset")
    ap.add_argument("-o",
                    "--output",
                    type=str,
                    default="output/",
                    help="path to output directory")
    ap.add_argument("-f",
                    "--face_detector",
                    type=str,
                    default="models/face_detector/",
                    help="path to face detector model")
    ap.add_argument("-m",
                    "--mask_detector",
                    type=str,
                    default="models/mask_detector/",
                    help="path to mask detector model")
    ap.add_argument("-c",
                    "--confidence",
                    type=float,
                    default=0.6,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    execute(args["testset"],
            args["output"],
            args["face_detector"],
            args["mask_detector"],
            args["confidence"])
