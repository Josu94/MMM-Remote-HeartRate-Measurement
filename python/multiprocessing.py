from multiprocessing import Lock
from multiprocessing import Queue
from multiprocessing import Process
import cv2
import dlib
import imutils


#
# Variables
#
imageWidth = 640
imageHeight = 480
frameRate = 20
recordingTime = 2 * frameRate


#
# Loading Dlib's face detector & facial landmark predictor (this takes a while, especially for the 68p predictor --> aprox. 100mb)
#
print('INFO: Load dlibs face detector.')
detector = dlib.get_frontal_face_detector()

print('INFO: Load dlibs face predictor (facial landmarks).')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


#
# Capture images and store them in a FIFO queue
#
def capture_images(queue, lock):
    global imageWidth
    global imageHeight
    global frameRate
    global recordingTime

    #
    # Setup the camera
    #
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight);
    cap.set(cv2.CAP_PROP_FPS, frameRate);
    if not cap.isOpened():
        cap.open()

    frameCounter = 0

    while frameCounter < recordingTime:
        success, frame = cap.read()
        if success:
            # Resize frame so save computation power
            frame = imutils.resize(frame, width=400)
            # Lock critical part (insert new data to the shared memory --> queue)
            lock.acquire()
            queue.put(frame)
            lock.release()
            # Increase frameCounter
            frameCounter += 1
            print(frameCounter)
        else:
            print('Capture stream lost...')

#
# Main process
#
if __name__ == '__main__':
    lock = Lock()
    q = Queue()
    capture_process = Process(target=capture_images, args=(q, lock))
    capture_process.start()
    # Wait until capture process has finished his work
    capture_process.join()
    # Program finished
    print('Wrote ' + recordingTime + ' frames to queue.')
