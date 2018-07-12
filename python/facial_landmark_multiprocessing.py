import os
import cv2
import time
import dlib
import imutils
import logging
import multiprocessing
import numpy as np
import queue                               # for catching queue.Empty exception in worker process
import csv
from multiprocessing import Lock, Queue, Process, Pool
from imutils import face_utils
from matplotlib.path import Path
from datetime import datetime

#
# Variables
#
imageWidth = 640
imageHeight = 480
frameRate = 20
recordingTime = 32 * frameRate
q = Queue()
timestamps = []

#
# Setup the camera
#
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight);
cap.set(cv2.CAP_PROP_FPS, frameRate);
if not cap.isOpened():
    cap.open()

#
# Loading Dlib's face detector & facial landmark predictor (this takes a while, especially for the 68p predictor --> aprox. 100mb)
#
# print('INFO: Load dlibs face detector.')
detector = dlib.get_frontal_face_detector()

# print('INFO: Load dlibs face predictor (facial landmarks).')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


#
# Capture images and store them in a FIFO queue
#
def capture_images():
    frameCounter = 0

    while True:
        if frameCounter < recordingTime:
            success, frame = cap.read()
            if success:
                # Resize frame so save computation power
                frame = imutils.resize(frame, width=400)
                q.put(frame)
                timestamps.append(datetime.utcnow())

                # Increase frameCounter
                frameCounter += 1
                #print(frameCounter)
            else:
                print('Capture stream lost...')
        else:
            # Cleanup camera object
            cap.release()
            break

    print('Finished capturing...')


#
# Process image in own process.
#
def process_image_worker(q, result):
    global detector
    global predictor

    # enable multithreading in OpenCV for child thread --> https://github.com/opencv/opencv/issues/5150
    #cv2.setNumThreads(-1)

    print(os.getpid(), "working")
    while True:
        # Measure processing time for each frame
        sw_start = time.time()

        try:
            frame = q.get(block=True, timeout=1)
        except queue.Empty:
            print('Queue is empty...')
            return

        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces and create bounding box
        rects = detector(gray, 0)

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # define the ROI from the given landmarks --> here: cheeks-area
            polygon = [(shape[1][0], shape[1][1]), (shape[2][0], shape[2][1]), (shape[3][0], shape[3][1]),
                       (shape[4][0], shape[4][1]), (shape[31][0], shape[31][1]), (shape[32][0], shape[32][1]),
                       (shape[33][0], shape[33][1]), (shape[34][0], shape[34][1]), (shape[35][0], shape[35][1]),
                       (shape[12][0], shape[12][1]), (shape[13][0], shape[13][1]), (shape[14][0], shape[14][1]),
                       (shape[15][0], shape[15][1]), (shape[28][0], shape[28][1])]

            poly_path = Path(polygon)

            x, y = np.mgrid[:gray.shape[0], :gray.shape[1]]
            x, y = x.flatten(), y.flatten()
            coors = np.vstack((y, x)).T

            mask = poly_path.contains_points(coors)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            # for (x, y) in shape:
            #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            mask = mask.reshape(gray.shape[0], gray.shape[1])

            # # get only masked pixel values in BGR format
            # roi_pixels = frame[mask == True]
            #
            # # get average BGR values from ROI --> Be careful with BRG format (pleace double check this!!!)
            # mean_bgr = np.array(roi_pixels).mean(axis=(0))
            # mean_r = mean_bgr[2]
            # mean_g = mean_bgr[1]
            # mean_b = mean_bgr[0]
            # print(mean_g)

            # get only masked pixel values in HSV format
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            roi_pixels = frame_hsv[mask == True]

            # hue = roi_pixels[:,0]
            # saturation = roi_pixels[:,1]
            # value = roi_pixels[:,2]
            #
            # color_mask = ((hue > 0) | (hue < 46)) & ((saturation > 23) | (saturation < 132)) & ((value > 88) | (value < 255))
            #
            # roi_pixels_refinement = roi_pixels[color_mask == True]

            # get average Hue value from ROI
            mean_hsv = np.array(roi_pixels).mean(axis=(0))
            mean_hue = mean_hsv[0]

            #result.put(mean_hue)
            if result:
                result.put(mean_hue)

            # color ROI in black
            #frame[mask] = 0

        # Measure processing time for each frame
        sw_stop = time.time()
        seconds = sw_stop - sw_start
        #print('Worker tooks %f seconds.' % seconds)
        print(seconds)


def drain(q):
    while True:
        try:
            yield q.get_nowait()
        except queue.Empty:  # on python 2 use Queue.Empty
            break

#
# Main process
#
if __name__ == '__main__':
    # disable multithreading in OpenCV for main thread to avoid problems after fork --> https://github.com/opencv/opencv/issues/5150
    cv2.setNumThreads(0)

    # Initialize Logger
    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)

    m = multiprocessing.Manager()
    result = m.Queue()

    # Create Pool of 3 worker processes
    pool = multiprocessing.Pool(3, process_image_worker, (q, result))

    # Start capture images
    capture_images()

    # Wait if all worker processes are done
    pool.close()
    pool.join()

    #
    # Save meanValue + corresponding timestamp into .csv file for displaying and compare ppg with ecg ground truth data
    #
    mean_values = []
    for item in drain(result):
        print(item)
        mean_values.append(item)
    # prepare measurements and save them to csv file
    output = zip(timestamps, mean_values)

    with open('TESTING/ppg_signal.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')

        # Zip the two lists and access pairs together.
        for item1, item2 in output:
            print(item1, "...", item2)
            writer.writerow((item1, item2))

    # Program finished
    print('Programm exit.')
