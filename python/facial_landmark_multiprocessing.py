#
# Imports
#
import os
import cv2
import time
import dlib
import imutils
import logging
import multiprocessing
import numpy as np
import queue  # for catching queue.Empty exception in worker process
import csv
import scipy.signal
import matplotlib.pyplot as plt
from multiprocessing import Lock, Queue, Process, Pool, Value
from imutils import face_utils
from matplotlib.path import Path
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy import fftpack

#
# Variables
#
imageWidth = 640
imageHeight = 480
frameRate = 20
recordingTime = 35 * frameRate
firstMeasurement = 30 * frameRate       # First measurement after 30 seconds
additionalMeasurement = 1 * frameRate   # Further update HR every second
q = Queue()
timestamps = []
frameCounter = None

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
print('Load dlibs face detector.')
detector = dlib.get_frontal_face_detector()

print('Load dlibs face predictor.')
predictor = dlib.shape_predictor('modules/MMM-Remote-HeartRate-Measurement/python/shape_predictor_68_face_landmarks.dat')
#predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



#
# Capture images and store them in a FIFO queue
#
def capture_images():
    global frameCounter

    while True:
        if frameCounter.value < recordingTime:
            success, frame = cap.read()
            if success:
                # Resize frame so save computation power
                frame = imutils.resize(frame, width=300)
                q.put(frame)
                timestamps.append(datetime.utcnow())

                # Increase frameCounter
                with frameCounter.get_lock():
                    frameCounter.value += 1
            #else:
                #print('Capture stream lost...')
        else:
            # Cleanup camera object
            cap.release()
            break

    #print('Finished capturing...')


#
# Process image in own process.
#
def process_image_worker(q, result):
    global detector
    global predictor

    # enable multithreading in OpenCV for child thread --> https://github.com/opencv/opencv/issues/5150
    # cv2.setNumThreads(-1)

    #print(os.getpid(), "working")
    while True:
        # Measure processing time for each frame
        sw_start = time.time()

        try:
            frame = q.get(block=True, timeout=1)
        except queue.Empty:
            #print('Queue is empty...')
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

            if result:
                result.put(mean_hue)

                # color ROI in black
                # frame[mask] = 0

        # Measure processing time for each frame
        sw_stop = time.time()
        seconds = sw_stop - sw_start
        # print('Worker tooks %f seconds.' % seconds)
        #print(seconds)


#
# Empty queue
#
def drain(q):
    while True:
        try:
            yield q.get_nowait()
        except queue.Empty:
            break


#
# Save meanValue + corresponding timestamp into .csv file for displaying and compare ppg with ecg ground truth data
#
def store_results():
    mean_values = []
    for item in drain(result):
        #print(item)
        mean_values.append(item)
    # prepare measurements and save them to csv file
    output = zip(timestamps, mean_values)

    with open('TESTING/ÖhlerJosua/Macbook/ppg_signal.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')

        # Zip the two lists and access pairs together.
        for item1, item2 in output:
            #print(item1, "...", item2)
            writer.writerow((item1, item2))


#
# Moving average filter
#
def moving_average(hue_values, window):
    weights = np.repeat(1.0, window) / window
    ma = np.convolve(hue_values, weights, 'valid')
    return ma


#
# Bandpass filter
#
def bandpass(data, lowcut, highcut, sr, order=5):
    passband = [lowcut * 2 / sr, highcut * 2 / sr]
    b, a = butter(order, passband, 'bandpass')
    y = filtfilt(b, a, data, axis=0, padlen=0)
    return y


#
# Fast Furier Transformation
#
def fft_transformation(signal):
    global frameRate
    time_step = 1 / frameRate  # in Hz

    # The FFT of the signal
    sig_fft = fftpack.fft(signal)

    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)

    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(signal.size, d=time_step)

    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

    # # Plot fft result
    # plt.figure(figsize=(8, 8))
    # plt.plot(sample_freq, power)
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('plower')
    # plt.show()

    return peak_freq


#
# Calculate the heartrate with fast furier transformation.
# In first iteration wait for 30 seconds of video material. After that refresh heartrate every second (sliding window of 30s)
#
def calculate_heartrate(result):
    global frameRate
    global frameCounter
    global firstMeasurement
    global additionalMeasurement

    # Show 1-3 dots in GUI to display progress
    progress = 0

    fft_window = []

    # Sample rate and desired cutoff frequencies (in Hz).
    sr = frameRate
    lowcut = 0.75
    highcut = 4.0

    # If inicialize_hr is True, wait for 30 seconds of video input from queue
    calculate_hr_started = False

    # Endless loop
    while True:
        # Count entries in result queue
        result_counter = result.qsize()

        if result_counter == firstMeasurement:
            # TODO: Berechne HR mit 30 s videomaterial
            data_counter = 0
            queue_result = 0
            while data_counter < firstMeasurement:
                try:
                    queue_result = result.get(block=True, timeout=3)
                except queue.Empty:
                    # print('Result-Queue is empty...')
                    return
                fft_window.append(queue_result)
                data_counter += 1

            # Apply Bandpass filter
            s1 = bandpass(fft_window, lowcut, highcut, sr, order=5)

            # Apply moving average filter
            s2 = moving_average(s1, 8)

            # Detrend signal
            s3 = scipy.signal.detrend(s2)

            # FFT transformation
            peak_freq = fft_transformation(s3)

            # Print out HR to the console
            #print('####################### First HR estimation...')
            print(round(peak_freq * 60, 0))

            calculate_hr_started = True

        elif calculate_hr_started == True and result_counter % additionalMeasurement == 0:
            # TODO: Berechne HR mit 30 s videomaterial (sliding window) --> davon sind Anzahl frameRate frames neu
            # Delete first N-elements in Array.
            fft_window = fft_window[additionalMeasurement:]
            # Append array with new values from queue (1 second of new data)
            data_counter = 0
            queue_result = 0
            while data_counter < additionalMeasurement:
                try:
                    queue_result = result.get(block=True, timeout=3)
                except queue.Empty:
                    # print('Result-Queue is empty...')
                    return
                fft_window.append(queue_result)
                data_counter += 1

            # Apply Bandpass filter
            s1 = bandpass(fft_window, lowcut, highcut, sr, order=5)

            # Apply moving average filter
            s2 = moving_average(s1, 8)

            # Detrend signal
            s3 = scipy.signal.detrend(s2)

            # FFT transformation
            peak_freq = fft_transformation(s3)

            # Print out HR to the console
            #print('####################### Updating HR estimation...')
            print(round(peak_freq * 60, 0))

        # Implementing Progress bar:
        if progress == 0:
            print('.')
            progress += 1
        elif progress == 10000:
            print('..')
            progress += 1
        elif progress == 20000:
            print('...')
            progress += 1
        elif progress == 30000:
            print('')
            progress = 0
        else:
            progress += 1


#
# Main process
#
if __name__ == '__main__':
    # Print text on MagicMirror
    print('Herzfrequenz wird gemessen...')

    # Disable multithreading in OpenCV for main thread to avoid problems after fork --> https://github.com/opencv/opencv/issues/5150
    cv2.setNumThreads(0)

    # Initialize a cross-process framecounter
    frameCounter = Value('i', 0)

    # Initialize Logger
    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)

    m = multiprocessing.Manager()
    result = m.Queue()

    # Create Pool of worker processes
    pool = multiprocessing.Pool(2, process_image_worker, (q, result))

    # Calculate heart rate with fft
    hr_estimation = Process(target=calculate_heartrate, args=(result,))
    hr_estimation.start()

    print('Start capturing frames...')
    # Start capture images
    capture_images()

    # Wait if all worker processes are done
    pool.close()
    pool.join()

    # Wait if last heart_rate estimation is finished
    # TODO: It could be, that there are not enought frames at the end for a new estimation. So we have to terminate this thread at this point!
    hr_estimation.join()

    # Store results
    # store_results()

    # Program finished
    #print('Programm exit.')

