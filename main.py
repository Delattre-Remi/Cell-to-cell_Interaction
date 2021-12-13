import cv2
import numpy as np

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
    for val in arr:
        if(val[0] >= value - thresh and val[0] <= value + thresh): 
            bool = True
            break
    return bool

def printArr(arr):
    """Print an array line by line

    Args:
        arr (array): Array to print
    """
    for x in range(len(arr[0])) : print(arr[x])

img = cv2.imread("Data/Brightfield/BF no cells.jpg")
img = rotate_image(img, 1.5)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,th = cv2.threshold(gray, 127, 255, 0)

# Trouver tous les contours trouvables sur l'image
contours, hierarchy = cv2.findContours(th,2,1)

# Filter seulement les contours qui nous intéressent (ceux entre 500 et 800 pixels de surface)
contours = list(filter(lambda a : cv2.contourArea(a) < 800 and cv2.contourArea(a) > 500 ,contours))

# Coefficient pour la taille des rectangles
size = 33
centers = []

# Créer un rectangle autour de chaque contour choisi
for c in contours : 
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centers.append((cX,cY))

centers.sort()

# Créer les colonnes de puits
last = centers[0][0]
cluster = []
clusters = []
for point in centers:
    if(np.abs(point[0] - last) < 50) : cluster.append(point)
    else:
        clusters.append(cluster)
        cluster = [point]
    last = point[0]
clusters.append(cluster)

# Créer les lignes noires qui relient les colonnes de puits
verticalClustersLengths = []
for cluster in clusters:
    cluster.sort(key = lambda a : a[1])
    verticalClustersLengths.append(len(cluster))
    cv2.line(img, cluster[0], cluster[-1], (0,0,0), thickness=3)

numberOfVertficalClusters = len(verticalClustersLengths)
numberOfColumnsOnOneSide = int(numberOfVertficalClusters / 2)
centerColumnIndex = numberOfColumnsOnOneSide
w, h1 = np.amax(verticalClustersLengths), numberOfColumnsOnOneSide
print("There is {} holes per column, {} column on each side and one in the center".format(w, h1))

# Colonne du centre
centerColumnX = int(np.mean(list(map(lambda a : a[0] ,clusters[centerColumnIndex]))))
cv2.line(img, (centerColumnX, 0), (centerColumnX, 2040), (255,255,0), thickness= 3)

# Création des rectanles autour des puits détectés via les contours
for center in centers:
    cX = center[0]
    cY = center[1]
    startPoint = (cX - size, cY - size)
    endPoint   = (cX + size, cY + (size * 2))
    width, height = endPoint[0] - startPoint[0], endPoint[1] - startPoint[1]
    center = (int(startPoint[0] + width/2), int(startPoint[1] + height/2))
    if(cX > centerColumnX + 10) : 
        startPoint = (startPoint[0] , startPoint[1] - size)
        endPoint   = (endPoint[0]   , endPoint[1]   - size)
    cv2.circle(img, center, 2, (0,0,255), thickness=5)
    cv2.rectangle(img, startPoint, endPoint, (255, 0, 0))

cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imwrite("recognition.png", img)

h = h1 * 2 + 1
presenceArray = [[0 for _ in range(w)] for _ in range(h)] 

printArr(presenceArray)