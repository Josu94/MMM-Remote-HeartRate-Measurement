# Imports
import cv2
import time
import threading
import argparse
import dlib
# import skin_pixel_detection
import numpy as np
import imutils
import csv
# import matplotlib.pyplot as plt
from matplotlib.path import Path
from multiprocessing import Queue
from imutils import face_utils
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from datetime import datetime

# Camera settings go here
imageWidth = 640
imageHeight = 480
#imageWidth = 1640
#imageHeight = 1232
frameRate = 20
processingThreads = 4
totalFrameNumber = 1200

temporal_stride = 20
global frameCounter
frameCounter = 0
# Make result global for plotting in main Thread
global result

# Shared values
global running
global cap
global frameLock
global processorPool
running = True
frameLock = threading.Lock()
global queue_frame
queue_frame = Queue()


# Setup the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight);
cap.set(cv2.CAP_PROP_FPS, frameRate);
if not cap.isOpened():
    cap.open()
# cap = VideoStream(usePiCamera=False, resolution=(imageWidth,imageHeight), framerate=frameRate).start()
time.sleep(1.0)

print('INFO: Load dlibs face detector.')
detector = dlib.get_frontal_face_detector()

print('INFO: Load dlibs face predictor (facial landmarks).')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# storage for mean values
mean_values = np.zeros(totalFrameNumber, dtype='float64')
timestamps = []


def spacial_subspace_rotation():
    # result (pulse signal)
    global result
    result = [totalFrameNumber]

    # eigenvalues and eigenvectors for each frame
    eigenvalues = np.zeros((totalFrameNumber, 3), dtype='float64')
    eigenvectors = np.zeros((totalFrameNumber, 3, 3), dtype='float64')

    processing_counter = 0
    while True:
        ############### Processing for each image in queue goes here #####################

        if not queue_frame.empty():
            # Get the last frame from the queue
            image = queue_frame.get()

            # Convert image to grayscale
            frame_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get bounding box arround the face
            global detector
            rects = detector(frame_grey, 0)
            for rect in rects:
                # compute the bounding box of the face
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)

                # Draw rectangle of BoundingBox to the image
                # cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

            # Crop the image
            crop_image = image[bY:bY + bH, bX:bX + bW]

            # Converting image into different colourspaces for calculating the skinpixel matrix1
            frame_argb = skin_pixel_detection.convert_bgr_to_argb(crop_image)
            frame_hsv = skin_pixel_detection.convert_bgr_to_hsv(crop_image)
            frame_ycbcr = skin_pixel_detection.convert_bgr_to_ycbcr(crop_image)

            # Calculate skinpixel matrix1
            skinpixel_matrix = skin_pixel_detection.get_skinpixel_matrix_1(crop_image, frame_argb, frame_hsv,
                                                                           frame_ycbcr, saveFrames=False)

            skinpixel_matrix = np.transpose(skinpixel_matrix)

            # Convert skinpixel matrix from RGB colorspace into HSV --> Extract Puls signal only from the Hue value
            # hue_array = skin_pixel_detection.get_hue_array(skinpixel_matrix)

            # Calculate mean value from hue_array
            # hue_mean_value = np.mean(hue_array)
            # print(hue_mean_value)
            # global hue_mean_array
            # hue_mean_array.append(hue_mean_value)

            # Get eigenvalues and eigenvectors
            eigenvalues[processing_counter], eigenvectors[
                processing_counter] = skin_pixel_detection.get_eigenvalues_and_eigenvectors(skinpixel_matrix)

            # Plot eigenvectors (it's only possible in the main Thread!)
            # skin_pixel_detection.plot_eigenvectors(skinpixel_matrix, eigenvectors)

            # Increase counter
            processing_counter = processing_counter + 1

            # Build P and add it to the puls signal if temporal stride has reached. Else increase processing_counter
            if processing_counter % temporal_stride == 0:
                global p
                p = skin_pixel_detection.build_p(processing_counter, temporal_stride, np.transpose(eigenvectors),
                                                 np.transpose(eigenvalues))
                # result[processing_counter] = p
                for n in p:
                    result.append(n)
                    print(n)


        else:
            print('LOG: FIFO Queue is empty')
            break


def facial_landmarks_plus_mean_hue():

    processing_counter = 0
    while True:
        ############### Processing for each image in queue goes here #####################

        if not queue_frame.empty():
            # start time for calculating FPS
            start = time.time()

            ###################################################################
            # find_bottleneck_start = time.time()
            # find_bottleneck_stop = time.time()
            # elapsed_time = find_bottleneck_stop - find_bottleneck_start
            ###################################################################

            # Get the last frame from the queue
            frame = queue_frame.get()
            frame = imutils.resize(frame, width=200)

            # convert image to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            find_bottleneck_start = time.time()

            # detect faces and create bounding box
            rects = detector(gray, 0)

            find_bottleneck_stop = time.time()
            elapsed_time = find_bottleneck_stop - find_bottleneck_start
            print('Elapsed time face-detection: %f' % elapsed_time)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            for rect in rects:
                find_bottleneck_start = time.time()

                shape = predictor(gray, rect)

                find_bottleneck_stop = time.time()
                elapsed_time = find_bottleneck_stop - find_bottleneck_start
                print('Elapsed time facial-landmark: %f' % elapsed_time)

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

                mean_values[processing_counter] = mean_hue

                # color ROI in black
                frame[mask] = 0

            # find_bottleneck_start = time.time()
            #
            # show the frame
            #cv2.imshow("Frame", frame)
            #key = cv2.waitKey(1) & 0xFF
            #
            # find_bottleneck_stop = time.time()
            # elapsed_time = find_bottleneck_stop - find_bottleneck_start
            # print('Elapsed time print frame: %f' % elapsed_time)

            # calculation for FPS
            stop = time.time()
            seconds = stop - start
            fps = 1 / seconds
            #print(fps)
            print('Total processing time: %f' % seconds)

            processing_counter += 1

        else:
            print('LOG: FIFO Queue is empty')
            break


# Image processing thread, self-starting
class ImageQueueing(threading.Thread):
    def __init__(self, name, autoRun=True):
        super(ImageQueueing, self).__init__()
        self.event = threading.Event()
        self.eventWait = (2.0 * processingThreads) / frameRate
        self.name = str(name)
        print('ImageQueuing thread %s started with idle time of %.2fs' % (self.name, self.eventWait))
        self.start()

    def run(self):
        # This method runs in a separate thread
        global running
        global frameLock
        global processorPool
        while running:
            # Wait for an image to be written to the stream
            self.event.wait(self.eventWait)
            if self.event.isSet():
                if not running:
                    break
                try:
                    self.queue_image(self.nextFrame)
                finally:
                    # Reset the event
                    self.nextFrame = None
                    self.event.clear()
                    # Return ourselves to the pool at the back
                    with frameLock:
                        processorPool.insert(0, self)
        print('ImageQueuing thread %s terminated' % (self.name))

    def queue_image(self, image):

        # put image to queue
        global queue_frame
        queue_frame.put(image)


# Image capture thread, self-starting
class ImageCapture(threading.Thread):
    def __init__(self):
        super(ImageCapture, self).__init__()
        self.start()

    # Stream delegation loop
    def run(self):
        # This method runs in a separate thread
        global running
        global cap
        global processorPool
        global frameLock
        global frameCounter
        while running:
            # Grab the oldest unused processor thread
            with frameLock:
                if processorPool:
                    processor = processorPool.pop()
                else:
                    processor = None
            if processor:
                if frameCounter <= totalFrameNumber:
                    # Grab the next frame and send it to the processor
                    success, frame = cap.read()
                    # frame = cap.read()
                    frameCounter = frameCounter + 1
                    if success:
                        # TODO: Create queue for timestamps
                        timestamps.append(datetime.utcnow())
                        processor.nextFrame = frame
                        processor.event.set()
                    else:
                        print('Capture stream lost...')
                        running = False
                else:
                    running = False
            else:
                # When the pool is starved we wait a while to allow a processor to finish
                time.sleep(0.01)
        print('Capture thread terminated')


# Create some threads for frame capturing and queuing
processorPool = [ImageQueueing(i + 1) for i in range(processingThreads)]
allProcessors = processorPool[:]
captureThread = ImageCapture()

################################################
# Queue Processing loop goes here (main Thread)#
################################################

# sleep for two seconds, that the queue can fill up, and then start processing the frames in the queue
time.sleep(15)

facial_landmarks_plus_mean_hue()

# prepare measurements and save them to csv file
output = zip(timestamps, mean_values)

with open('TESTING/ppg_signal.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')

    # Zip the two lists and access pairs together.
    for item1, item2 in output:
        print(item1, "...", item2)
        writer.writerow((item1, item2))



# Cleanup all processing threads
running = False
while allProcessors:
    # Get the next running thread
    with frameLock:
        processor = allProcessors.pop()
    # Send an event and wait until it finishes
    processor.event.set()
    processor.join()

# Cleanup the capture thread
captureThread.join()

# Cleanup the camera object
cap.release()

cv2.destroyAllWindows()
