# Imports
import cv2
import time
import threading
import argparse
import dlib
from imutils import face_utils
from imutils.video import VideoStream
from multiprocessing import Queue

# Camera settings go here
imageWidth = 640
imageHeight = 480
frameRate = 20
processingThreads = 4
processingFrames = 40
global frameCounter
frameCounter = 0

# Shared values
global running
global cap
global frameLock
global processorPool
running = True
frameLock = threading.Lock()
global queue
queue = Queue()

# Setup the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight);
cap.set(cv2.CAP_PROP_FPS, frameRate);
if not cap.isOpened():
    cap.open()
# cap = VideoStream(usePiCamera=False, resolution=(imageWidth,imageHeight), framerate=frameRate).start()
time.sleep(1.0)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


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
                    self.queue_image(self.nextFrame, self.nextFrameName)
                finally:
                    # Reset the event
                    self.nextFrame = None
                    self.event.clear()
                    # Return ourselves to the pool at the back
                    with frameLock:
                        processorPool.insert(0, self)
        print('Processor thread %s terminated' % (self.name))

    def queue_image(self, image, name):

        # First put image to queue then the Name (timestamp) for the image
        global queue
        queue.put(image)


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
                if frameCounter <= processingFrames:
                    # Grab the next frame and send it to the processor
                    success, frame = cap.read()
                    # frame = cap.read()
                    frameCounter = frameCounter + 1
                    if success:
                        processor.nextFrame = frame
                        processor.nextFrameName = 'video/image%10.4f.jpg' % time.time()
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
        time.sleep(2)
        while True:
            ############### Processing for each image in queue goes here #####################

            if not queue.empty():
                image = queue.get()
                # Find face in frame and compute facial landmarks
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces in the grayscale frame
                global detector
                rects = detector(gray_image, 0)

                # loop over the face detections
                global predictor
                for rect in rects:
                    # compute the bounding box of the face and draw it on the
                    # frame
                    (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                    cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH),
                                  (0, 255, 0), 1)

                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(gray_image, rect)
                    shape = face_utils.shape_to_np(shape)

                    # loop over the (x, y)-coordinates for the facial landmarks
                    # and draw each of them
                    for (i, (x, y)) in enumerate(shape):
                        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                        # print('Nr.%d: x:%d, y:%d' % (i, x, y))
                        cv2.putText(image, str(i + 1), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                # Dump ndarray frames to disk as .jpg picture
                fileName = 'video/image%10.4f.jpg' % time.time()
                cv2.imwrite(filename=fileName, img=image)

            else:
                print('LOG: FIFO Queue is empty')
                break

            ##################################################################################
        print('QueueProcessing thread terminated')


# Create some threads for processing and frame grabbing
processorPool = [ImageQueueing(i + 1) for i in range(processingThreads)]
allProcessors = processorPool[:]
captureThread = ImageCapture()
queueProcessingThread = QueueProcessing(queue)

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
