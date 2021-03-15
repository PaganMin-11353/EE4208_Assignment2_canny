from PIL.Image import new
import cv2
from time import time
import numpy as np
import os
import matplotlib.pyplot as plt


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


def canny(img,kernel):
    # 5 steps
    # 1.Noise reduction (guassian blur)
    # img_gaussian = convolution(img, kernel_gaussian)
    img_gaussian = convolution(img, kernel)
    cv2.imshow('gaussian_blur',img_gaussian.astype(np.uint8))
    
    # 2.edge enhancement (gradient caluclation)
    
    cv2.imshow('edge enhancement',img_enhance)
    result = img_enhance
    # 3.Non-maximum suppression;

    # 4.Double threshold;

    # 5.Edge Tracking by Hysteresis.
    return result

def read():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_ext = '.png'
    image_name = 'lena'
    image_path = os.path.join(base_dir, image_name + image_ext)
    image = cv2.imread(image_path)
    if len(image.shape) == 3:
        # RGB weight: L = R * 299/1000 + G * 587/1000 + B * 114/1000
        rgb_weights = [0.2989, 0.587, 0.114]
        gray_image = np.dot(image[...,:3], rgb_weights) # normalize the grayimage to 0.0 to 1.0 to display
        gray_image = gray_image.astype(np.uint8)
    elif len(image.shape) == 2:
        gray_image = image

    # gray_image /= 255. # normalize the grayimage to 0.0 to 1.0 to display
    # new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(new_gray)
    # print(gray_image) # opencv image is brighter, probabaly due to the low precision for fast computation speed

    # CANNY
    gaussian_kernel = generate_gaussian(size=5)
    canny_image = canny(gray_image, gaussian_kernel)

    cv2.imshow('original', image)
    cv2.imshow('gray', gray_image)
    cv2.imshow('canny', canny_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
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
    read()
    # print(generate_gaussian(5))
    # gaussian separable. use 1D filter to reduce calculation time
    # print(cv2.getGaussianKernel(ksize=5,sigma=1) * cv2.getGaussianKernel(ksize=5,sigma=1).T)