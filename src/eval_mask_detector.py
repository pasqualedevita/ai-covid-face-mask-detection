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


def get_csv_path(path):
    print("[INFO] loading csv ...")
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csvPath = os.path.sep.join([path, file])
    return csvPath


def get_image_path(path, image):
    print("[INFO] loading image ...")
    for file in os.listdir(path):
        if file.startswith(image):
            imagePath = os.path.sep.join([path, file])
    return imagePath


def execute_mask_detector(input_csv_file, input_dir, output_dir, face_detector, mask_detector, confidence):
    clean_dir(output_dir)
    output_results = []
    with open(input_csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            input_image_name = row[0]
            input_with_mask_count = row[1]
            input_without_mask_count = row[2]
            input_with_incorrect_mask_count = row[3]
            if input_image_name == 'image':
                # skip header row
                continue

            image_path = get_image_path(input_dir, input_image_name)
            image_mask_detector_result = image_mask_detector.execute(
                image_path,
                face_detector,
                mask_detector,
                confidence)
            cv2.imwrite(output_dir+os.path.basename(image_path),
                        image_mask_detector_result[0])
            output_results.append([input_image_name,
                                   image_mask_detector_result])

    return output_results


def save_results(mask_detector_results, output_dir):
    with open(output_dir+'data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["image", "with_mask", "without_mask", "with_incorrect_mask"])
        for mask_detector_result in mask_detector_results:
            image = mask_detector_result[0]
            with_mask = 0
            without_mask = 0
            with_incorrect_mask = 0
            for result in mask_detector_result[1][1]:
                if result[0] == 'with_mask':
                    with_mask = with_mask + 1
                if result[0] == 'without_mask':
                    without_mask = without_mask + 1
                if result[0] == 'with_incorrect_mask':
                    with_incorrect_mask = with_incorrect_mask + 1
            writer.writerow([image,
                             with_mask,
                             without_mask,
                             with_incorrect_mask])


def eval_accuracy(input_csv_file, ouput_csv_file, output_dir):
    input_rows = []
    output_rows = []

    with open(input_csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'image':
                # skip header row
                continue
            input_rows.append(row)

    with open(ouput_csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'image':
                # skip header row
                continue
            output_rows.append(row)

    count_images = len(input_rows)
    count_images_ok = 0
    eval_count = range(0, count_images)
    for i in eval_count:
        if input_rows[i][1] != output_rows[i][1]:
            continue
        if input_rows[i][2] != output_rows[i][2]:
            continue
        if input_rows[i][3] != output_rows[i][3]:
            continue
        count_images_ok = count_images_ok + 1

    accuracy = count_images_ok / count_images
    with open(output_dir+'accuracy.txt', 'w') as output:
        output.write(str(accuracy))
    return accuracy


def execute(input_dir='testset/input/', output_dir='testset/output/', face_detector='models/face_detector/', mask_detector='models/mask_detector/', confidence=0.6):
    input_csv_file = get_csv_path(input_dir)
    mask_detector_results = execute_mask_detector(input_csv_file,
                                                  input_dir,
                                                  output_dir,
                                                  face_detector,
                                                  mask_detector,
                                                  confidence)
    save_results(mask_detector_results, output_dir)
    output_csv_file = get_csv_path(output_dir)
    eval_accuracy(input_csv_file, output_csv_file, output_dir)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",
                    "--input_dir",
                    type=str,
                    default="testset/input/",
                    help="path to input directory")
    ap.add_argument("-o",
                    "--output_dir",
                    type=str,
                    default="testset/output/",
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

    execute(args["input_dir"],
            args["output_dir"],
            args["face_detector"],
            args["mask_detector"],
            args["confidence"])
