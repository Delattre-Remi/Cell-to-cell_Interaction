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

def isSameColor(c1, c2):
    return (c1[0] == c2[0] and c1[1] == c2[1] and c1[2] == c2[2])

def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

CENTER_COLOR = (255,0,255)

img = cv2.imread("Data/Brightfield/BF no cells.jpg")
img = rotate_image(img, 1.5)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,th = cv2.threshold(gray, 127, 255, 0)

# Trouver tous les contours trouvables sur l'image
contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter seulement les contours qui nous intéressent (ceux entre 500 et 800 pixels de surface)
contours = list(filter(lambda a : cv2.contourArea(a) < 800 and cv2.contourArea(a) > 500 ,contours))
cv2.drawContours(img, contours, -1, (0,255,0), 3)

# Coefficient pour la taille des rectangles
size = 33
centers = []

# Créer un rectangle autour de chaque contour choisi
for c in contours : 
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centers.append((cX,cY))

# Créer les colonnes de puits
centers.sort()
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
for cluster in clusters:
    cluster.sort(key = lambda a : a[1])
    cv2.line(img, cluster[0], cluster[-1], (0,0,0), thickness=3)

numberOfVertficalClusters = len(clusters)
numberOfColumnsOnOneSide = int(numberOfVertficalClusters / 2)
centerColumnIndex = numberOfColumnsOnOneSide
w, h1 = np.amax(np.amax([len(x) for x in clusters])), numberOfColumnsOnOneSide
print("There is {} holes per column, {} column on each side and one in the center".format(w, h1))

# Colonne du centre
centerColumnX = int(np.mean(list(map(lambda a : a[0] ,clusters[centerColumnIndex]))))
cv2.line(img, (centerColumnX, 0), (centerColumnX, 2040), (255,255,0), thickness= 3)

# Création des rectangles autour des puits détectés via les contours
for center in centers:
    cX, cY = center[0], center[1]
    startPoint = (cX - size, cY - size)
    endPoint   = (cX + size, cY + (size * 2))
    width  = endPoint[0] - startPoint[0]
    height = endPoint[1] - startPoint[1]
    center = (int(startPoint[0] + width/2), int(startPoint[1] + height/2))
    if(cX > centerColumnX + 10) : 
        startPoint = (startPoint[0] , startPoint[1] - size)
        endPoint   = (endPoint[0]   , endPoint[1]   - size)
    cv2.rectangle(img, startPoint, endPoint, (255, 0, 0))
    cv2.circle(img, center, 2, (0,0,255), thickness=5)
    cv2.circle(img, center, 0, (0,255,255), thickness=2)
    cv2.circle(img, center, 0, CENTER_COLOR)

cv2.imwrite("recognition.png", img)
horizontalClusters = []
centersToCheck = centers
cluster = []
maxDistance = distance((0,0), (2 * width, 2 * height)) * 1.2

found = False
horizontalClustersLengths = []
while (len(centersToCheck) > 0) :
    if(not found) : 
        center = centersToCheck[0]
        centersToCheck.remove(center)
        horizontalClusters.append(cluster)
        horizontalClustersLengths.append(len(cluster))
        cluster = [center]
    
    # Esquiver la colonne centrale
    if(center[0] >= centerColumnX - 10 and center[0] <= centerColumnX + 10) : continue

    # Choisir s'il s'agit de la gauche ou de la droite que l'on analyse
    if(center[0] < centerColumnX) : heightRange = range(int(height * 0.9), int(height * 1.5))
    else : heightRange =  range(int(-height * 0.90), int(-height * 0.40))

    found = False
    for j in heightRange:
        y = center[1] + j
        if(y > 2047) : continue
        for i in range(int(1.2 * width), int(1.6 * width)):
            x = center[0] + i 
            if(x > 2047) : continue
            if(isSameColor(img[y][x], CENTER_COLOR)) : 
                closestCenter = (x, y - 16)
                centersToCheck.remove(closestCenter)
                center = closestCenter
                cluster.append(closestCenter)
                found = True               
                break
        if found : break

# Créer les lignes vertes qui relient les lignes de puits
for cluster in horizontalClusters:
    if(len(cluster) > 0) : cv2.line(img, cluster[0], cluster[-1], (0,255,0), thickness=3)

print(np.amax([len(x) for x in horizontalClusters]))

h = h1 * 2 + 1
presenceArray = [[0 for _ in range(w)] for _ in range(h)] 
printArr(presenceArray)
cv2.imwrite("recognition2.png", img)