# Imports
import cv2
import time
import threading
import argparse
import dlib
import skin_pixel_detection
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Queue
from imutils import face_utils
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from datetime import datetime

# Camera settings go here
imageWidth = 640
imageHeight = 480
frameRate = 20
processingThreads = 4
totalFrameNumber = 100
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
global queue_timestamps
queue_timestamps = Queue()
global hue_mean_array
hue_mean_array = []

# Setup the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight);
cap.set(cv2.CAP_PROP_FPS, frameRate);
if not cap.isOpened():
    cap.open()
# cap = VideoStream(usePiCamera=False, resolution=(imageWidth,imageHeight), framerate=frameRate).start()
time.sleep(1.0)


detector = skin_pixel_detection.detector
print('INFO: Load dlibs face detector.')
#predictor = skin_pixel_detection.predictor
predictor = dlib.shape_predictor('../shape_predictor_5_face_landmarks.dat')
print('INFO: Load dlibs face predictor (facial landmarks).')

# storage for mean values
mean_values = np.zeros(1250, dtype='float64')
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
    # result (pulse signal)
    global result
    result = [totalFrameNumber]

    processing_counter = 0
    while True:
        ############### Processing for each image in queue goes here #####################

        if not queue_frame.empty():
            # Get the last frame from the queue
            image = queue_frame.get()

            # Convert image to grayscale
            frame_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # # Get bounding box arround the face
            # global detector
            # rects = detector(frame_grey, 0)
            # for rect in rects:
            #     # compute the bounding box of the face
            #     (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            #
            #     # Draw rectangle of BoundingBox to the image
            #     # cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)


            #rects = skin_pixel_detection.get_face_bounding_box()
            # Get facial landmarks
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rect = detector(gray, 0)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Draw facial landmarks on image
            for (i, (x, y)) in enumerate(shape):
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                # print('Nr.%d: x:%d, y:%d' % (i, x, y))
                cv2.putText(image, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            ################################################################################################
            ################################################################################################
            ################################################################################################

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
        print('Processor thread %s started with idle time of %.2fs' % (self.name, self.eventWait))
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
        print('Processor thread %s terminated' % (self.name))

    def queue_image(self, image):

        # First put image to queue then the Name (timestamp) for the image
        global queue_frame
        global queue_timestamps
        queue_frame.put(image)
        queue_timestamps.put(datetime.utcnow())


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


# FIFO queue processing Thread, self-starting
class QueueProcessing(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.daemon = True
        self.queue = queue
        self.start()

    # Check if queue has entries, if yes process them
    def run(self):
        # Wait a short time, so that the queue contains some frames to be processed
        time.sleep(2)

        # Using 2sr method to extract heartbeat information from frames in queue
        #spacial_subspace_rotation()

        # Using facial landmarks to gernerate ROI and extract the mean hue value out of it.
        facial_landmarks_plus_mean_hue()

        print('QueueProcessing thread terminated')


# Butterworth filter
def butter_bandpass(lowcut, highcut, fs, order=9):
    # Nyquist-Frequenz
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# Butterworth filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=9):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Moving average filter
def moving_average(hue_values, window):
    weights = np.repeat(1.0, window) / window
    ma = np.convolve(hue_values, weights, 'valid')
    return ma


# Create some threads for processing and frame grabbing
processorPool = [ImageQueueing(i + 1) for i in range(processingThreads)]
allProcessors = processorPool[:]
captureThread = ImageCapture()
queueProcessingThread = QueueProcessing(queue_frame)

# Main loop, basically waits until you press CTRL+C
# The captureThread gets the frames and passes them to an unused processing thread
try:
    print('Press CTRL+C to quit')
    while running:
        time.sleep(1)
except KeyboardInterrupt:
    print('\nUser shutdown')
except:
    e = sys.exc_info()
    print(e)
    print('\nUnexpected error, shutting down!')

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

# Cleanup the queueProcessing thread
queueProcessingThread.join()

# Cleanup the camera object
cap.release()
