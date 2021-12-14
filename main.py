import cv2
import numpy as np

# region UtilFunctions

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

def printArr(arr):
    """Print an array line by line

    Args:
        arr (array): Array to print
    """
    for x in range(len(arr[0])) : print(arr[x])

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
        if(distance(point, p) < maxDist and p != point) : return True
    return False

# endregion UtilFunctions

# region InitConstants
CENTER_COLOR = (255,0,255)
CALCULATED = 50
DETECTED = 0
IMG_WIDTH = 2048
IMG_HEIGHT = 2048
RECTANGLE_SIZE = 33
PROSSESED_DIRNAME = "Processed/"
DEBUG = True

# endregion InitConstants

# region CustomFunctions

def getContourCenters(img, draw = False):
    if(DEBUG) : test_img = cv2.imread(PROSSESED_DIRNAME + "rotated.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray, 127, 255, 0)

    # Trouver tous les contours trouvables sur l'image
    contours, _ = cv2.findContours(th,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter seulement les contours qui nous intéressent (ceux entre 500 et 800 pixels de surface)
    contours = list(filter(lambda a : cv2.contourArea(a) < 800 and cv2.contourArea(a) > 500 ,contours))
    if(draw) : cv2.drawContours(img, contours, -1, (0,255,0), 3)

    # Coefficient pour la taille des rectangles
    centers = []

    for c in contours : 
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Si le centre est à gauche du mileu de l'image, offset vers le bas pour avoir le centre du puit
        # Vers le haut pour la droite du mileu
        if(cX > IMG_WIDTH / 2 + 50) : cY -= 20
        else                        : cY += 20
        if(DEBUG) : cv2.circle(test_img, (cX, cY), 5, (255,0,0), thickness=3)
        if(DEBUG) : cv2.circle(test_img, (cX, cY), 0, CENTER_COLOR)
        cv2.circle(img, (cX, cY), 5, (255,0,0), thickness=3)
        cv2.circle(img, (cX, cY), 0, CENTER_COLOR)
        centers.append((cX,cY))

    if(DEBUG) : cv2.imwrite(PROSSESED_DIRNAME + "Step_1_marked_centers.png", test_img)
    centers.sort()
    return centers

def getVerticalClusters(centers):
    last = centers[0][0]
    cluster = []
    verticalClusters = []
    # Pour chaque point, si le précédent est assez proche horizontalement, 
    # ils sont alignés verticalement,
    # et font donc partir de la même colonne
    for point in centers:
        if(np.abs(point[0] - last) < 50) : cluster.append(point)
        else:
            verticalClusters.append(cluster)
            cluster = [point]
        last = point[0]
    verticalClusters.append(cluster)
    return verticalClusters

def getHorizontalClusters(centersToCheck, numberOfColumnsOnOneSide, centerColumnX):
    if(DEBUG) : test_img = cv2.imread(PROSSESED_DIRNAME + "Step_1_marked_centers.png")
    found = False
    cluster = []
    horizontalClusters = []
    centerColumn = []
    size = RECTANGLE_SIZE * 2
    # Groupement des points horizontalement
    while (len(centersToCheck) > 0) :
        if(not found) : 
            center = centersToCheck[0]
            centersToCheck.remove(center)
            horizontalClusters.append(cluster)
            cluster = [center]

        # Esquiver la colonne centrale
        if(center[0] >= centerColumnX - 10 and center[0] <= centerColumnX + 10) : 
            centerColumn.append((numberOfColumnsOnOneSide, center, DETECTED))
            continue

        # Choisir s'il s'agit de la gauche ou de la droite que l'on analyse
        if(center[0] < centerColumnX) : heightRange = range(int( size * 1.10), int( size * 1.35))
        else                          : heightRange = range(int(-size * 1.40), int(-size))

        found = False
        for j in heightRange:
            y = center[1] + j
            if(y > IMG_WIDTH - 1) : continue
            for i in range(int(1.3 * size), int(1.55 * size)):
                x = center[0] + i 
                if(x > IMG_HEIGHT - 1) : continue
                if(DEBUG and test_img[y][x][0] != 255) : test_img[y][x] = (0,255,0)
                if(isSameColor(img[y][x], CENTER_COLOR)) : 
                    closestCenter = (x, y)
                    centersToCheck.remove(closestCenter)
                    cluster.append(closestCenter)
                    center = closestCenter
                    found = True               
                    break
            if found : break

    horizontalClusters.pop(0)
    if(DEBUG) : 
        for cluster in horizontalClusters:
            cv2.line(test_img, cluster[0], cluster[-1], (0,0,0))
        cv2.imwrite(PROSSESED_DIRNAME + "Step_2_detection.png", test_img)
    return [horizontalClusters, centerColumn]

def getXposOfColumns(verticalClusters):
    xPositionOfColumns = []
    for cluster in verticalClusters:
        xPositionOfColumns.append(list(map(lambda a : a[0], cluster)))
    return [int(np.mean(x)) for x in xPositionOfColumns]

def associateHorizontalClusterWithXpos(horizontalClusters, XposOfColumns):
    if(DEBUG) : test_img = cv2.imread(PROSSESED_DIRNAME + "Step_1_marked_centers.png")
    horizontalClustersWithX = []
    for cluster in horizontalClusters:
        tmpCluster = []
        for point in cluster:
            found, index = hasValueBetween(XposOfColumns, point[0], 25)
            if(found) : 
                tmpCluster.append([index, point, DETECTED])
                if(DEBUG) : cv2.putText(test_img, str(index), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        horizontalClustersWithX.append(tmpCluster)
    if(DEBUG) : cv2.imwrite(PROSSESED_DIRNAME + "Step_3_xpos.png", test_img)
    return horizontalClustersWithX

def filterHorizontalClusters(horizontalClustersWithX):
    leftIncompleteLines = []
    rightIncompleteLines = []
    leftCompleteLines = []
    rightCompleteLines = []
    centerCluster = []
    for cluster in horizontalClustersWithX:
        if(cluster[0][0] == 0 and cluster[-1][0] == numberOfColumnsOnOneSide - 1):
            # Ligne complète à gauche
            leftCompleteLines.append(cluster)
        elif(cluster[0][0] == numberOfColumnsOnOneSide + 1 and cluster[-1][0] == 2 * numberOfColumnsOnOneSide):
            # Ligne complète à droite
            rightCompleteLines.append(cluster)
        else :
            # Ligne incomplète
            if(cluster[0][1][0] > centerColumnX - 20 and cluster[0][1][0] < centerColumnX + 20) : centerCluster.append(cluster)  
            elif(cluster[0][1][0] < centerColumnX) : leftIncompleteLines.append(cluster)
            else                                   : rightIncompleteLines.append(cluster)
    return [leftIncompleteLines, leftCompleteLines, rightIncompleteLines, rightCompleteLines]

def getOffsets(leftCompleteLines, rightCompleteLines):
    xOffsetSum = 0
    yOffsetSum = 0
    x = 0
    for cluster in leftCompleteLines:
        for i in range(len(cluster) - 1):
            xOffsetSum += (cluster[i + 1][1][0] - cluster[i][1][0])
            yOffsetSum += (cluster[i + 1][1][1] - cluster[i][1][1])
            x += 1

    leftxOffset = int(xOffsetSum / x)
    leftyOffset = int(yOffsetSum / x)

    xOffsetSum = 0
    yOffsetSum = 0
    x = 0
    for cluster in rightCompleteLines:
        for i in range(len(cluster) - 1):
            xOffsetSum += (cluster[i + 1][1][0] - cluster[i][1][0])
            yOffsetSum += (cluster[i + 1][1][1] - cluster[i][1][1])
            x += 1

    rightxOffset = int(xOffsetSum / x)
    rightyOffset = int(yOffsetSum / x)
    return [leftxOffset, leftyOffset, rightxOffset, rightyOffset]

def completeHorizontalClusters(leftIncompleteLines, rightIncompleteLines, leftxOffset, leftyOffset, rightxOffset, rightyOffset):
    if(DEBUG) : test_img = cv2.imread(PROSSESED_DIRNAME + "Step_1_marked_centers.png")
    leftCompletedLines = []
    existingPoints = [x[1] for list in leftIncompleteLines for x in list]
    for cluster in leftIncompleteLines:
        firstIndex = cluster[0][0]
        lastIndex = cluster[-1][0]
        while(firstIndex > 0) :
            pos = (cluster[0][1][0] - leftxOffset, cluster[0][1][1] - leftyOffset) 
            cluster.insert(0, [firstIndex - 1, pos, CALCULATED])
            if(not isCloseToOtherPointInArr(existingPoints, pos, 50)) : cluster.append([lastIndex + 1, pos, CALCULATED])
            existingPoints.append(pos)
            firstIndex = cluster[0][0]
        while(lastIndex < numberOfColumnsOnOneSide - 1) : 
            pos = (cluster[-1][1][0] + leftxOffset, cluster[-1][1][1] + leftyOffset)
            cluster.append([lastIndex + 1, pos, CALCULATED])
            if(not isCloseToOtherPointInArr(existingPoints, pos, 50)) : cluster.append([lastIndex + 1, pos, CALCULATED])
            existingPoints.append(pos)
            lastIndex = cluster[-1][0]
        leftCompletedLines.append(cluster)

    rightCompletedLines = []
    existingPoints = [x[1] for list in rightIncompleteLines for x in list]
    for cluster in rightIncompleteLines:
        firstIndex = cluster[0][0]
        lastIndex = cluster[-1][0]
        while(firstIndex > numberOfColumnsOnOneSide + 1) :
            pos = (cluster[0][1][0] - rightxOffset, cluster[0][1][1] - rightyOffset)
            if(not isCloseToOtherPointInArr(existingPoints, pos, 50)) : cluster.insert(0, [firstIndex - 1, pos, CALCULATED])
            existingPoints.append(pos)
            firstIndex -= 1
        while(lastIndex < 2 * numberOfColumnsOnOneSide) : 
            pos = (cluster[-1][1][0] + rightxOffset, cluster[-1][1][1] + rightyOffset)
            if(not isCloseToOtherPointInArr(existingPoints, pos, 50)) : cluster.append([lastIndex + 1, pos, CALCULATED])
            existingPoints.append(pos)
            lastIndex += 1
        rightCompletedLines.append(cluster)
    if(DEBUG) : 
        for line in leftCompletedLines + rightCompletedLines:
            for point in line:
                if(point[2] == CALCULATED) : cv2.circle(test_img, point[1], 4,(0,255,0), thickness=3)
        cv2.imwrite(PROSSESED_DIRNAME + "Step_4_completion.png", test_img)
    return [leftCompletedLines, rightCompletedLines]

# endregion CustomFunctions

img = cv2.imread("Data/Brightfield/BF no cells.jpg")
img = rotate_image(img, 1.5)
cv2.imwrite(PROSSESED_DIRNAME + "rotated.png", img)

# On récupère le centre de chaque rectangles formé autour des contours détectés par OpenCV
centers = getContourCenters(img)

# On regroupe chaque centre dans une colonne
verticalClusters = getVerticalClusters(centers)
numberOfColumnsOnOneSide = int(len(verticalClusters) / 2)

# Marquage de la colonne du centre
centerColumnX = int(np.mean(list(map(lambda a : a[0], verticalClusters[numberOfColumnsOnOneSide]))))
cv2.line(img, (centerColumnX, 0), (centerColumnX, 2040), (255,255,0), thickness= 3)

# On regroupe chaque centre dans une ligne
horizontalClusters, centerColumn = getHorizontalClusters(centers, numberOfColumnsOnOneSide, centerColumnX)

# On calcules quelles abscisses correspondent à quelle colonne
XposOfColumns = getXposOfColumns(verticalClusters)

# On associe chaque centre de chaque ligne à sa colonne
horizontalClustersWithX = associateHorizontalClusterWithXpos(horizontalClusters, XposOfColumns)

# On peut alors identifier quelles lignes sont incomplètes et lesquelles le sont
leftIncompleteLines, leftCompleteLines, rightIncompleteLines, rightCompleteLines = filterHorizontalClusters(horizontalClustersWithX)
lineClusters = [leftIncompleteLines, rightIncompleteLines, leftCompleteLines, rightCompleteLines]

# On calcule l'écart moyen entre chaque centre pour l'appliquer aux centres manquants
leftxOffset, leftyOffset, rightxOffset, rightyOffset = getOffsets(leftCompleteLines, rightCompleteLines)

# On complète les centres que nous n'avions pas pu détecter
leftCompletedLines, rightCompletedLines = completeHorizontalClusters(leftIncompleteLines, rightIncompleteLines, leftxOffset, leftyOffset, rightxOffset, rightyOffset)

# On regroupe puis applati les points pour pouvoir les utiliser plus simplement
leftLines = leftCompletedLines + leftCompleteLines
rightLines = rightCompletedLines + rightCompleteLines
allCenters = [x for list in (leftLines + rightLines) for x in list] + centerColumn

# On trace un rectangle et un chiffre par point
img = cv2.imread(PROSSESED_DIRNAME + "rotated.png")
for point in allCenters:
    cX, cY = point[1][0], point[1][1]
    startPoint = (cX - RECTANGLE_SIZE, cY - RECTANGLE_SIZE)
    endPoint   = (cX + RECTANGLE_SIZE, cY + (RECTANGLE_SIZE * 2))
    if(cX > centerColumnX + 20) :
        startPoint = (startPoint[0], startPoint[1] - RECTANGLE_SIZE) 
        endPoint   = (endPoint[0]  , endPoint[1]   - RECTANGLE_SIZE) 
    width  = endPoint[0] - startPoint[0]
    height = endPoint[1] - startPoint[1]
    cv2.rectangle(img, startPoint, endPoint, (255,0,255))
    cv2.putText(img, str(point[0]), point[1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    if(point[2] == CALCULATED) : circle_color = (0,255,0)
    else                       : circle_color = (255,0,0)
    cv2.circle(img, point[1], 3, circle_color, thickness=3)

w = np.amax(np.amax([len(x) for x in verticalClusters]))
h = numberOfColumnsOnOneSide * 2 + 1
presenceArray = [[0 for _ in range(h)] for _ in range(w)] 

#printArr(presenceArray)
cv2.imwrite("recognition.png", img)