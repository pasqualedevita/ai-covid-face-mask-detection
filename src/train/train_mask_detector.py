# USAGE
# python src/train/train_mask_detector.py

# import the necessary packages
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d",
                "--dataset",
                default="dataset/",
                type=str,
                help="path to input dataset")
ap.add_argument("-m",
                "--model",
                type=str,
                default="models/mask_detector/",
                help="path to output mask detector model")
ap.add_argument("-i",
                "--ignored_dataset",
                type=str,
                default="no_ignore",
                help="list of ignored dataset separated by ,")
args = vars(ap.parse_args())

dataset_dir = args["dataset"]
model_dir = args["model"]
ignored_dataset = args["ignored_dataset"].split(',')

# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 1e-4
EPOCHS = 5
BS = 32

# train test size and random state
TRAIN_TEST_SIZE = 0.25
TRAIN_TEST_RANDOM_STATE = 42

# head model parameters
HEAD_MODEL_POOL_SIZE = (7, 7)
HEAD_MODEL_DENSE = 256
HEAD_MODEL_DROPOUT = 0.25

# grab the list of images in our dataset directory, then initialize the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = []
data = []
labels = []
valid_images = [".jpg", ".jpeg", ".png"]
for dirname, dirs, filenames in os.walk(dataset_dir, topdown=True):
    for filename in filenames:
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_images:
            continue
        imagePaths.append(os.path.join(dirname, filename))

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # jump ignored dataset
    if label in ignored_dataset:
        continue

    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
unique_labels = np.unique(labels)

# perform one-hot encoding on the labels
if len(unique_labels) == 2:
    lb = LabelBinarizer()
else:
    lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 1-TRAIN_TEST_SIZE the data for training and the remaining TRAIN_TEST_SIZE for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=TRAIN_TEST_SIZE,
                                                  stratify=labels,
                                                  random_state=TRAIN_TEST_RANDOM_STATE)

# data augmentation to construct the training image generator
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet",
                        include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=HEAD_MODEL_POOL_SIZE)(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(HEAD_MODEL_DENSE, activation="relu")(headModel)
headModel = Dropout(HEAD_MODEL_DROPOUT)(headModel)
headModel = Dense(len(unique_labels), activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# perform one-hot encoding on the labels
if len(unique_labels) == 2:
    loss = "binary_crossentropy"
else:
    loss = "categorical_crossentropy"
model.compile(loss=loss,
              optimizer=opt,
              metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
              steps_per_epoch=len(trainX) // BS,
              validation_data=(testX, testY),
              validation_steps=len(testX) // BS,
              epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1),
                            predIdxs,
                            target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
if (model_dir != ""):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
model.save(model_dir+"mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(model_dir+"mask_detector.png")