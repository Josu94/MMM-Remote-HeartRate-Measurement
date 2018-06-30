# Imports
import cv2 as cv
import numpy as np
import time
import cv2
import dlib
from scipy import misc
from imutils import face_utils

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_5_face_landmarks.dat')


# Convert BGR into ARGB format
def convert_bgr_to_argb(frame):
    # Convert BGR to ARGB with built-in opencv function
    rgba = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    # Create new np array with same size and type as rgba array. To do it simple, just copy the rgba np.array
    argb = np.copy(rgba)
    # Switch the alpha channel to index 0
    cv.mixChannels(rgba, argb, [0,1, 1,2, 2,3, 3,0])
    return argb


# Convert BGR into HSV format
def convert_bgr_to_hsv(frame):
    # Convert BGR to HSV with built-in opencv function
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return hsv


# Convert BGR into YCbCr format
def convert_bgr_to_ycbcr(frame):
    # Convert BGR to YCbCr with built-in opencv function
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    return hsv


# Calculate skinpixel matrix (boolean values), based on predefined rules: https://arxiv.org/pdf/1708.02694.pdf
def get_skinpixel_matrix_1(image, argb, hsv, ycbcr, saveFrames=False):
    # Create skinmask (in a new nparray, label each pixel with True or False)
    skinmask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

    # Slice HSV array
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    # Slice ARGB array
    r = argb[:, :, 0]
    g = argb[:, :, 1]
    b = argb[:, :, 2]
    a = argb[:, :, 3]

    # Slice YCbCr array
    y_ = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 2]
    cr = ycbcr[:, :, 1]

    # Loop over skinmask matrix:
    for (x, y), value in np.ndenumerate(skinmask):
        h_value = h[x][y]
        s_value = s[x][y]
        r_value = r[x][y]
        g_value = g[x][y]
        b_value = b[x][y]
        a_value = a[x][y]
        y_value = y_[x][y]
        cb_value = cb[x][y]
        cr_value = cr[x][y]

        # Info: S-Wert von HSV wird normal in Prozent angegeben. Der Wert wurde aus diesem Grund hier x 100 genommen
        if 0 <= h_value <= 50 and 23 <= s_value <= 68 and r_value > 95 and g_value > 40 and b_value > 20 and r_value > g_value and r_value > b_value and (
                    r_value - g_value) > 15 and a_value > 15 \
                or r_value > 95 and g_value > 40 and b_value > 20 and r_value > g_value and r_value > b_value and (
                            r_value - g_value) > 15 and a_value > 15 and cr_value > 135 and cb_value > 85 and y_value > 80 \
                        and cr_value <= ((1.5862 * cb_value) + 20) and cr_value >= (
                            (0.3448 * cb_value) + 76.2069) and cr_value >= (
                            (-4.5652 * cb_value) + 234.5652) and cr_value <= (
                            (-1.15 * cb_value) + 301.75) and cr_value <= ((-2.2857 * cb_value) + 432.85):

            # if r_value > 95 and g_value > 40 and b_value > 20 and r_value > g_value and r_value > b_value and (
            #             r_value - g_value) > 15 and a_value > 15 and cr_value > 135 and cb_value > 85 and y_value > 80 \
            #                 and cr_value <= ((1.5862 * cb_value) + 20) and cr_value >= (
            #             (0.3448 * cb_value) + 76.2069) and cr_value >= ((-4.5652 * cb_value) + 234.5652) and cr_value <= (
            #             (-1.15 * cb_value) + 301.75) and cr_value <= ((-2.2857 * cb_value) + 432.85):

            # if 0 <= h_value <= 50 and 23 <= s_value <= 68 and r_value > 95 and g_value > 40 and b_value > 20 and r_value > g_value and r_value > b_value and (
            #             r_value - g_value) > 15 and a_value > 15:

            skinmask[x][y] = True

        else:
            skinmask[x][y] = False

    # Change Axis from image
    image_ = image.transpose((-1, 0, 1))

    # Save masked face
    # masked_image = image
    # masked_image[skinmask] = 0
    # misc.imsave('maskedImage/face%10.4f.png' % time.time(), masked_image)

    # Draw the result on the screen
    # from scipy import misc
    # fileName = 'maskedImage/face_%10.4f.jpg' % time.time()
    # cv.imwrite(filename=fileName, img=image)
    #
    # counter = 0
    # for (x, y, z), value in np.ndenumerate(image):
    #     if skinmask[x][y] == False:
    #         image[x][y] = 0
    #         counter = counter + 1
    #
    # fileName = 'maskedImage/face%10.4f.jpg' % time.time()
    # cv.imwrite(filename=fileName, img=image)

    # Get skinpixel Matrix
    skin_pixels = image_[:, skinmask]
    skin_pixels = np.transpose(skin_pixels)

    skin_pixels = skin_pixels.astype('float64') / 255.0

    # print('INFO: Calculation skinpixel matrix finished...')
    return skin_pixels


# Convert skinpixel matrix with shape (3, n) from RGB into single H-values
def get_hue_array(skinpixels):
    # create a new numpy array for storing the single H values of the skinpixels
    hue_array = np.zeros(skinpixels.shape[1], dtype='uint8')
    # initialise counter variable
    i = 0

    while i < skinpixels.shape[1]:
        H = None

        R_ = skinpixels[0][i] / 255
        G_ = skinpixels[1][i] / 255
        B_ = skinpixels[2][i] / 255

        c_max = max(R_, G_, B_)
        c_min = min(R_, G_, B_)
        delta = c_max - c_min

        # Hue clalculation
        if c_max == c_min:
            H = 0

        elif c_max == R_:
            H = 60 * (0 + ((G_ - B_) / (c_max - c_min)))

        elif c_max == G_:
            H = 60 * (2 + ((B_ - R_) / (c_max - c_min)))

        elif c_max == B_:
            H = 60 * (4 + ((R_ - G_) / (c_max - c_min)))

        if H < 0:
            H = H + 360

        hue_array[i] = H
        i += 1
        # print(H)
    return hue_array


# Build C-Matrix, get eigenvalue and eigenvectors, sort them
def get_eigenvalues_and_eigenvectors(skin_pixels):
    # build the correlation matrix
    c = np.dot(skin_pixels, skin_pixels.T)
    c /= skin_pixels.shape[1]

    # get eigenvectors and sort them according to eigenvalues (largest first)
    evals, evecs = np.linalg.eig(c)
    idx = evals.argsort()[::-1]
    eigenvalues = evals[idx]
    eigenvectors = evecs[:, idx]
    return eigenvalues, eigenvectors


# Plots skin pixel cluster and eignevectors in the RGB space
def plot_eigenvectors(skin_pixels, eigenvectors):
    origin = np.array([0, 0, 0])
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(skin_pixels[0], skin_pixels[1], skin_pixels[2])
    ax.plot([origin[0], eigenvectors[0, 0]], [origin[1], eigenvectors[1, 0]], zs=[origin[2], eigenvectors[2, 0]],
            color='g')
    for k in range(1, 3, 1):
        ax.plot([origin[0], eigenvectors[k, 0]], [origin[1], eigenvectors[k, 1]], zs=[origin[2], eigenvectors[k, 2]],
                color='r')
    plt.show()


# Build P-Signal
def build_p(counter, temporal_stride, eigenvectors, eigenvalues, plot=False):
    """build_P(counter, temporal_stride, eigenvectors, eigenvalues, plot=False) -> p
  
      Builds P
  
      **Parameters**
  
        ``counter`` (int):
          The frame index
  
        ``temporal_stride`` (int):
          The temporal stride to use
  
        ``eigenvectors`` (numpy array):
          The eigenvectors of the c matrix (for all frames up to counter). 
  
        ``eigenvalues`` (numpy array):
          The eigenvalues of the c matrix (for all frames up to counter).
  
        ``plot`` (boolean):
          If you want something to be plotted
  
      **Returns**
  
        ``p`` (numpy array):
          The p signal to add to the pulse.
    """
    tau = counter - temporal_stride

    # SR'
    sr_prime_vec = np.zeros((3, temporal_stride), 'float64')
    c2 = 0
    for t in range(tau, counter, 1):
        # equation 11
        sr_prime = np.sqrt(eigenvalues[0, t] / eigenvalues[1, tau]) * np.dot(eigenvectors[:, 0, t].T,
                                                                                   np.outer(eigenvectors[:, 1, tau],
                                                                                               eigenvectors[:, 1,
                                                                                               tau].T))
        sr_prime += np.sqrt(eigenvalues[0, t] / eigenvalues[2, tau]) * np.dot(eigenvectors[:, 0, t].T,
                                                                                    np.outer(eigenvectors[:, 2, tau],
                                                                                                eigenvectors[:, 2,
                                                                                                tau].T))
        sr_prime_vec[:, c2] = sr_prime
        c2 += 1

    # build p and add it to the final pulse signal (equation 12 and 13)
    p = sr_prime_vec[0, :] - ((np.std(sr_prime_vec[0, :]) / np.std(sr_prime_vec[1, :])) * sr_prime_vec[1, :])

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(temporal_stride), sr_prime_vec[0, :], c='b')
        ax.plot(range(temporal_stride), sr_prime_vec[1, :], c='b')
        ax.plot(range(temporal_stride), p, c='r')
        plt.show()

    return p


# Compute bounding box from face of the given frame. Returns bounding Box.
def get_face_bounding_box(image):
    # convert image to grey image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_image, 0)

    for rect in rects:
        # compute the bounding box of the face and draw it on the frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)

        # Draw rectangle of BoundingBox to the image
        # cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

    return rects


# Compute facial landmarks in the given frame. Returns nparray.
def get_facial_landmarks(image):
    global predictor
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = detector(gray_image, 0)
    shape = predictor(gray_image, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw each of them
    # for (i, (x, y)) in enumerate(shape):
    #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    #     # print('Nr.%d: x:%d, y:%d' % (i, x, y))
    #     cv2.putText(image, str(i + 1), (x - 10, y - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Dump ndarray frames to disk as .jpg picture
    # fileName = 'video/image%10.4f.jpg' % time.time()
    # cv2.imwrite(filename=fileName, img=image)

    return shape


