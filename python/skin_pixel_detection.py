import cv2 as cv
import numpy as np

# Convert BGR into ARGB format
def convert_bgr_to_argb(frame):
    # Convert BGR to ARGB with built-in opencv function
    rgba = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    # Create new np array with same size and type as rgba array. To do it simple, just copy the rgba np.array
    argb = np.copy(rgba)
    # Switch the alpha channel to index 0
    cv.mixChannels(rgba, argb, [0, 1, 1, 2, 2, 3, 3, 0])
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
def get_skinpixel_matrix_1(argb, hsv, ycbcr):
    # Create skinpixel matrix
    skinpixel_matrix = np.zeros((argb.shape[0], argb.shape[1]), dtype=bool)

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

    # Loop over skinpixel matrix:
    # skinpixel_matrix[719][1279] = 1
    for (x, y), value in np.ndenumerate(skinpixel_matrix):
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

        #if r_value > 95 and g_value > 40 and b_value > 20 and r_value > g_value and r_value > b_value and (
        #             r_value - g_value) > 15 and a_value > 15 and cr_value > 135 and cb_value > 85 and y_value > 80 \
        #                 and cr_value <= ((1.5862 * cb_value) + 20) and cr_value >= (
        #             (0.3448 * cb_value) + 76.2069) and cr_value >= ((-4.5652 * cb_value) + 234.5652) and cr_value <= (
        #             (-1.15 * cb_value) + 301.75) and cr_value <= ((-2.2857 * cb_value) + 432.85):

        #if 0 <= h_value <= 50 and 23 <= s_value <= 68 and r_value > 95 and g_value > 40 and b_value > 20 and r_value > g_value and r_value > b_value and (
        #             r_value - g_value) > 15 and a_value > 15:

            skinpixel_matrix[x][y] = True
        else:
            skinpixel_matrix[x][y] = False

    print('INFO: Calculation skinpixel matrix ready...')
    return skinpixel_matrix
