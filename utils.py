import cv2
import numpy as np

DEBUG = True
PROSSESED_DIRNAME = "Processed/"
RECTANGLE_SIZE = 33

class PathError(Exception):
    pass

def load_image(imgPath, angle = 0):
    """Loads an image, rotates it if needed and does sanity checks

    Args:
        imgPath (String): Path to the image
        angle (int, optional): Angle the image needs to be rotated to. Defaults to 0.
    """
    img = cv2.imread(imgPath)
    if(img is None) : raise PathError('Given path {} is invalid !'.format(imgPath))
    return rotate_image(img, angle)

def rotate_image(image, angle):
    """Rotate the image by a certain angle

    Args:
        image (openCV image): The image to be rotated
        angle (int): The angle to rotate the image

    Returns:
        image: Rotated image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def hasValueBetween(arr, value, thresh):
    """Return if the array has a value close to the value given

    Args:
        arr (array): Array to scan through
        value (int): Value to look for
        thresh (int): Threshold to look around

    Returns:
        [bool]: True if found, else either way
    """
    bool = False
    for i in range(len(arr)):
        if(arr[i] >= value - thresh and arr[i] <= value + thresh): 
            bool = True
            break
    return [bool, i]

def printPositionArray(arr):
    y = 0
    str = ''
    for i in range(19) : str += "   |     {:2d}    ".format(i)
    print(str)
    for line in arr:
        strLine = "{:2d} | ".format(y)
        y += 1
        for point in line:
            if(type(point) == type(0)) : point = (-1,-1)
            strLine += "({:4d}, {:4d}) : ".format(point[0], point[1])
        print(strLine)

def isSameColor(c1, c2):
    return (c1[0] == c2[0] and c1[1] == c2[1] and c1[2] == c2[2])

def distance(a, b):
    return np.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))

def isCloseToOtherPointInArr(arr, point, maxDist):
    """Return true if any point in the given array is within maxDist to the point given

    Args:
        arr (array): array to look throught
        point (tuple): point to look around
        maxDist (int): maxDistance to look around

    Returns:
        [bool]: True if found a point False otherwise
    """ 
    for p in arr:
        first = True
        if(distance(point, p) < maxDist and (p != point and first)) : 
            first = False
            return True
    return False

def removeDuplicates(arr, maxDist):
    for point in arr:
        for p in arr:
            if(distance(point[1], p[1]) < maxDist and (arr.count(p) > 1 or p[0] != point[0])) : 
                arr.remove(p)
    return arr
