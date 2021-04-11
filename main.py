from operator import imatmul
from PIL.Image import new
import cv2
from time import time
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient


def convolve2D(image, kernel, padding=2, strides=1):
    # Cross Correlation, turn the kernel by 180 degrees
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(yImgShape):
        # Exit Convolution
        if y > yImgShape - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(xImgShape):
                # Go to next row once kernel is out of bounds
                if x > xImgShape - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return output

def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    #padding
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output

def generate_gaussian(size, sigma=1):
    size = int(size) // 2  # floor divition operator '//' true divitrion operator '/'
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    result = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return result

def generate_sobel(direction):
    if direction.lower() == 'x':
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    elif direction.lower() == 'y':
        return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

def RGB2GRAY(image):
    # RGB weight: L = R * 299/1000 + G * 587/1000 + B * 114/1000
    rgb_weights = [0.2989, 0.587, 0.114]
    result = np.dot(image[...,:3], rgb_weights) # normalize the grayimage to 0.0 to 1.0 to display
    result = result.astype(np.uint8)
    return result

def gaussian_blur(image):
    kernel = generate_gaussian(size=5)
    result = convolution(image, kernel)
    return result

def sobel_dege_detector(image):
    x_kernel = generate_sobel('x')
    y_kernel = generate_sobel('y')
    Ix = convolution(image, x_kernel)
    Iy = convolution(image, y_kernel)
    gradient_magnitude = np.sqrt(np.square(Ix) + np.square(Iy))
    gradient_direction = np.arctan2(Iy, Ix)
    return gradient_direction, gradient_magnitude, Ix, Iy

def nms(magnitude, dx, dy):
    # use sub-pixel interpolation to determine the edge pixel
    mag = np.copy(magnitude)
    M, N = np.shape(mag)
    result = np.zeros((M,N))

    for i in range(1, M-1):
        for j in range(1, N-1):
            if mag[i, j] == 0:
                result[i, j] =0
            else:
                gradX = dx[i,j]
                gradY = dy[i,j]
                gradX_abs = np.abs(gradX)
                gradY_abs = np.abs(gradY)
                grad = mag[i,j]

                # Y gradient greater than X gradient
                if gradY_abs > gradY_abs:
                    if gradY == 0:
                        weight = 0
                    else:
                        weight = gradX_abs / gradY_abs
                    # |g1|g2|  |    |  |g2|g1|
                    # |  | g|  | or |  | g|  |
                    # |  |g4|g3|    |g3|g4|  |
                    g2 = mag[i-1, j]
                    g4 = mag[i+1, j]
                    if gradX * gradY > 0:
                        g1 = mag[i-1, j-1]
                        g3 = mag[i+1, j+1]
                    else:
                        g1 = mag[i-1, j+1]
                        g3 = mag[i+1, j-1]
                # X gradient greater than Y gradient
                else:
                    if gradX == 0:
                        weight = 0
                    else:
                        weight = gradY_abs / gradX_abs
                    # |  |  |g3|    |g1|  |  |
                    # |g2| g|g4| or |g2| g|g4|
                    # |g1|  |  |    |  |  |g3|
                    g2 = mag[i, j-1]
                    g4 = mag[i, j+1]
                    if gradX * gradY > 0:
                        g1 = mag[i+1, j-1]
                        g3 = mag[i-1, j+1]
                    else:
                        g1 = mag[i-1, j-1]
                        g3 = mag[i+1, j+1]
                
                gradTemp1 = weight * g1 + (1-weight) * g2
                gradTemp2 = weight * g3 + (1-weight) * g4
                if grad >= gradTemp1 and grad >= gradTemp2:
                    result[i,j] = grad
                else:
                    result[i,j] = 0
    return result

def double_threshold(image, th_low_ratio = 0.1, th_high_ratio = 0.3):
    # upper lower ratio is recommended to be between 2:1 or 3:1
    highThreshold = np.max(image) * th_high_ratio
    lowThreshold = np.max(image) * th_low_ratio
    
    M, N = image.shape
    result = np.zeros((M,N), dtype=np.int32)
    
    weak_value = np.int32(50)
    strong_value = np.int32(255)
    
    strong_i, strong_j = np.where(image >= highThreshold)
    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
    # zeros_i, zeros_j = np.where(image < lowThreshold)
    
    result[strong_i, strong_j] = strong_value
    result[weak_i, weak_j] = weak_value

    return (result, weak_value)

def hysterisis(image, weak_value, strong_value=255):
    M, N = image.shape
    # result = np.zeros((M,N))
    top2btm = np.copy(image)
    right2left = np.copy(image)
    left2right = np.copy(image)

    # probably needs to go from other directions?
    for i in range(1, M):
        for j in range(1, N):
            # check the 8 surroundings of the weak edge
            if(top2btm[i,j]==weak_value):
                if((top2btm[i-1, j-1:j+1] == strong_value).any()
                     or (top2btm[i, [j-1,j+1]] == strong_value).any()
                     or (top2btm[i+1, j-1:j+1] == strong_value).any()):
                    top2btm[i,j] = strong_value
                else:
                    top2btm[i,j] = 0

    for i in range(1, M):
        for j in range(N-1, 0, -1):
            if(right2left[i,j]==weak_value):
                if((right2left[i-1, j-1:j+1] == strong_value).any()
                    or (right2left[i, [j-1,j+1]] == strong_value).any()
                    or (right2left[i+1, j-1:j+1] == strong_value).any()):
                    right2left[i,j] = strong_value
                else:
                    right2left[i,j] = 0

    for i in range(M-1, 0, -1):
        for j in range(1, N):
            if(left2right[i,j]==weak_value):
                if((left2right[i-1, j-1:j+1] == strong_value).any()
                    or (left2right[i, [j-1,j+1]] == strong_value).any()
                    or (left2right[i+1, j-1:j+1] == strong_value).any()):
                    left2right[i,j] = strong_value
                else:
                    left2right[i,j] = 0

    result = top2btm + right2left + left2right
    result[result > 255] = 255
    return result

def fit_parabola(x1, y1, x2, y2, x3, y3):
    # y = ax^2+bx+c
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    a = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    b = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    c = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
    x0 = (-1*b)/(2*a)
    return x0

def canny(image):
    # 5 steps
    # 1.Noise reduction (guassian blur)
    img_gaussian = gaussian_blur(image)
    cv2.imshow('gaussian_blur',img_gaussian.astype(np.uint8))
    
    # 2.edge enhancement (gradient caluclation)
    gradient_direction, gradient_magnitude, Ix, Iy = sobel_dege_detector(img_gaussian)
    cv2.imshow('edge enhancement',gradient_magnitude.astype(np.uint8))

    # 3.Non-maximum suppression (pixel accuracy)
    img_nms = nms(gradient_magnitude,dx=Ix, dy=Iy)
    cv2.imshow('NMS_result', img_nms.astype(np.uint8))

    # 4.Double thresholding
    img_dt, weak_value = double_threshold(img_nms)
    cv2.imshow('double thresholding', img_dt.astype(np.uint8))

    # 5.Edge Tracking by Hysteresis
    final_result = hysterisis(img_dt, weak_value=weak_value)
    cv2.imshow('hysterisis', final_result.astype(np.uint8))

    return None

def read_image(image_name, image_ext):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, image_name + '.' + image_ext)
    image = cv2.imread(image_path)
    if len(image.shape) == 3:
        gray_image = RGB2GRAY(image)
    elif len(image.shape) == 2:
        gray_image = image

    # # normalize the grayimage to 0.0 to 1.0 for display
    # gray_image /= 255. 
    # new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(new_gray)
    # # opencv image is brighter, probabaly due to the low precision for fast computation speed
    # print(gray_image) 

    # CANNY
    canny_image = canny(gray_image)

    cv2.imshow('original', image)
    cv2.imshow('gray', gray_image)
    # cv2.imshow('canny', canny_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def realtime():
    camera = cv2.VideoCapture(0)
    kernel_gauss = generate_gaussian(5)
    while True:
        # Read the frame
        ret, frame= camera.read()
        # mirroing the image
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = canny(gray,kernel_gauss)
        # canny_cv =  cv2.Canny(np.uint8(frame),200, 300)
        # Display
        cv2.imshow('canny', img)
        cv2.imshow('frame', frame)
        #cv2.imshow('canny_opencv', canny_cv)
        # Stop if escape key is pressed
        k = cv2.waitKey(20) & 0xff
        if k==27:
            break

if __name__ == '__main__':
    read_image('chessboard', 'jpg')
    # print(generate_gaussian(5))
    # gaussian separable. use 1D filter to reduce calculation time
    # print(cv2.getGaussianKernel(ksize=5,sigma=1) * cv2.getGaussianKernel(ksize=5,sigma=1).T)''