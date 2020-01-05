import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from PIL import Image
from tensorflow import one_hot
import constants


def apply_transformations(np_img, threshold_type):
    blurred = cv2.GaussianBlur(np_img, (5, 5), 0)
    thresholded = cv2.threshold(
        blurred, 0, 255, threshold_type+cv2.THRESH_OTSU)[1]
    result = cv2.morphologyEx(
        thresholded, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    return result


def load_training_data(train_dataset):
    x_train = []
    y_train = []
    train_dataset = glob(constants.TRAIN_PATH + '/*.png')

    for img in train_dataset:
        y_train.append(int(img[-6]))
        np_img = np.array(Image.open(img), dtype='uint8')
        np_img = apply_transformations(np_img, cv2.THRESH_BINARY)
        np_img = np.reshape(np_img, (128, 128, 1))
        x_train.append(np_img)
    x_train, y_train = np.array(x_train), np.array(y_train)
    print("Training data loaded")
    return (x_train, y_train)


def load_testing_data(test_dataset):
    x_test = []
    y_test = []
    test_dataset = glob(constants.TEST_PATH + '/*png')

    for img in test_dataset:
        y_test.append(int(img[-6]))
        np_img = np.array(Image.open(img), dtype='uint8')
        np_img = apply_transformations(np_img, cv2.THRESH_BINARY)
        np_img = np.reshape(np_img, (128, 128, 1))
        x_test.append(np_img)
    x_test, y_test = np.array(x_test), np.array(y_test)
    print("Testing data loaded")
    return (x_test, y_test)


def preprocess_data(x_train, x_test, y_train, y_test):
    classes = len(set(y_train))
    x_train, x_test = x_train/x_train.max(), x_test/x_test.max()
    y_train, y_test = one_hot(y_train, classes), one_hot(y_test, classes)
    print("Training and Testing data preprocessed")
    return x_train, x_test, y_train, y_test, classes


def preprocess_image(image):
    roi = image[constants.TOP:constants.BOTTOM, constants.RIGHT:constants.LEFT]
    roi = cv2.resize(roi, constants.IMG_DIM, interpolation=cv2.INTER_LINEAR)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    result_roi = apply_transformations(gray_roi, cv2.THRESH_BINARY_INV)
    cv2.imshow("ROI", result_roi)
    np_img = np.reshape(result_roi, constants.INPUT_DIM)
    np_img = np_img/np_img.max()
    return np_img


def display_frame(frame, label, score):
    cv2.putText(frame, "Count: " + str(math.floor(label)), constants.COUNT_POS,
                constants.FONT, constants.FONT_SCALE, constants.FONT_COLOR, constants.FONT_THICKNESS)
    cv2.putText(frame, "Accuracy: " + str(math.floor(score)), constants.ACCURACY_POS,
                constants.FONT, constants.FONT_SCALE, constants.FONT_COLOR, constants.FONT_THICKNESS)
    cv2.rectangle(frame, (constants.LEFT, constants.TOP),
                  (constants.RIGHT, constants.BOTTOM), (0, 255, 0), 2)
    cv2.imshow("Video Feed", frame)


def plot_metrics(r):
    plt.subplot(121)
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()

    plt.show()
