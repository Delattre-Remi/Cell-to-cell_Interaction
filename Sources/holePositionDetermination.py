import cv2
import numpy as np
from utils import *
from angleDetermination import getBestAngle

# Developped by https://github.com/Mostuniqu3/

# region InitConstants
CENTER_COLOR = (255,0,255)
CALCULATED = 50 # Identifiant d'un centre calculé
DETECTED = 0 # Identifiant d'un centre detecté
SHOW_CONTOURS = False
SHOW_LABELS = False
# endregion InitConstants

# region CustomFunctions

def getContourCenters(img, IMG_WIDTH):
    if(DEBUG) : test_img = cv2.imread(PROSSESED_DIRNAME + "rotated.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray, 127, 255, 0)

    # Trouver tous les contours trouvables sur l'image
    contours, _ = cv2.findContours(th,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter seulement les contours qui nous intéressent (ceux entre 500 et 800 pixels de surface)
    contours = list(filter(lambda a : cv2.contourArea(a) < 800 and cv2.contourArea(a) > 500 ,contours))
    if(SHOW_CONTOURS) : cv2.drawContours(img, contours, -1, (0,255,0), 3)

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
        if(DEBUG) : 
            cv2.circle(test_img, (cX, cY), 5, (255,0,0), thickness=3)
            cv2.circle(test_img, (cX, cY), 0, CENTER_COLOR)
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

def getHorizontalClusters(centersToCheck, numberOfColumnsOnOneSide, centerColumnX, centers, IMG_WIDTH, IMG_HEIGHT):
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
                if((x, y) in centers) : 
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

def filterHorizontalClusters(horizontalClustersWithX, numberOfColumnsOnOneSide, centerColumnX):
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

def completeHorizontalClusters(leftIncompleteLines, rightIncompleteLines, leftxOffset, leftyOffset, rightxOffset, rightyOffset, numberOfColumnsOnOneSide):
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
        cv2.imwrite(PROSSESED_DIRNAME + "Step_4_line_completion.png", test_img)
    return [leftCompletedLines, rightCompletedLines]

def fillMissingCenterColumn(allCenters, numberOfColumnsOnOneSide, centerColumn, verticalClusters) : 
    centersSortedByHeight = allCenters

    valuesSet = set([x[0] for x in centersSortedByHeight])
    centersSortedByHeight = [[list for list in centersSortedByHeight if list[0] == value] for value in valuesSet]
    centersSortedByHeight.sort(key = lambda x : x[0])
    centersSortedByHeight.sort(key = lambda x : x[1][1])

    upperPoint = centersSortedByHeight[numberOfColumnsOnOneSide][0][1][0]
    lowerPoint = centersSortedByHeight[numberOfColumnsOnOneSide + 1][1][1][0]
    found = False
    missingAtTop = 0
    while(not found) :
        for center in centerColumn:
            if(center[1][0] > upperPoint and center[1][0] < lowerPoint) : found = True
            else : 
                missingAtTop += 1
                upperPoint = centersSortedByHeight[numberOfColumnsOnOneSide + missingAtTop][0][1][0]
                lowerPoint = centersSortedByHeight[numberOfColumnsOnOneSide + missingAtTop + 1][1][1][0]

    w = np.amax(np.amax([len(x) for x in verticalClusters]))
    missingAtBottom = w - len(centerColumn)
    centerColumn.sort(key = lambda x : x[1][1])

    yOffsetSum = 0
    x = 0
    for i in range(len(centerColumn) - 1) :
        yOffsetSum += centerColumn[i + 1][1][1] - centerColumn[i][1][1]
        x += 1

    yOffset = int(yOffsetSum/x)

    while(missingAtBottom > 0):
        pos = (centerColumn[-1][1][0], centerColumn[-1][1][1] + yOffset)
        newPoint = [numberOfColumnsOnOneSide, pos, CALCULATED]
        centerColumnPos = [x[1] for x in centerColumn]
        if(not isCloseToOtherPointInArr(centerColumnPos, pos, 10)) :
            allCenters.append(newPoint)
            missingAtBottom -= 1
        centerColumn.append(newPoint)

    return [allCenters, w, centersSortedByHeight, yOffset]

def fillMissingHorizontalClusters(centersSortedByHeight, allCenters, numberOfColumnsOnOneSide, yOffset, w) : 
    valuesSet = range(0, 2 * numberOfColumnsOnOneSide + 1)
    firstLeftColumn = centersSortedByHeight[0]
    lastLeftColumn = centersSortedByHeight[numberOfColumnsOnOneSide - 1]
    while(len(firstLeftColumn) < w and len(lastLeftColumn) < w):
        for heightCluster in centersSortedByHeight : heightCluster.sort(key = lambda x : x[1][1])
        for i in range(numberOfColumnsOnOneSide):
            centerAbove = centersSortedByHeight[i][-1][1]
            pos = (centerAbove[0], centerAbove[1] + yOffset)
            newPoint = (i, pos, CALCULATED)
            allCenters.append(newPoint)
        centersSortedByHeight = allCenters
        centersSortedByHeight = [[list for list in centersSortedByHeight if list[0] == value] for value in valuesSet]
        centersSortedByHeight.sort(key = lambda x : x[1][1])
        firstLeftColumn = centersSortedByHeight[0]
        lastLeftColumn = centersSortedByHeight[numberOfColumnsOnOneSide - 1]

    firstRightColumn = centersSortedByHeight[numberOfColumnsOnOneSide + 1]
    lastRightColumn = centersSortedByHeight[-1]
    while(len(firstRightColumn) < w and len(lastRightColumn) < w):
        for heightCluster in centersSortedByHeight : heightCluster.sort(key = lambda x : x[1][1])
        for i in range(numberOfColumnsOnOneSide):
            centerAbove = centersSortedByHeight[i + numberOfColumnsOnOneSide + 1][-1][1]
            pos = (centerAbove[0], centerAbove[1] + yOffset)
            newPoint = [numberOfColumnsOnOneSide + 1 + i, pos, CALCULATED]
            allCenters.append(newPoint)
        centersSortedByHeight = allCenters
        centersSortedByHeight = [[list for list in centersSortedByHeight if list[0] == value] for value in valuesSet]
        centersSortedByHeight.sort(key = lambda x : x[1][1])
        firstRightColumn = centersSortedByHeight[numberOfColumnsOnOneSide + 1]
        lastRightColumn = centersSortedByHeight[-1]
    return allCenters

def populateArray(allCenters, numberOfColumnsOnOneSide) : 
    centersSortedByHeight = [[list for list in allCenters if list[0] == value] for value in range(2 * numberOfColumnsOnOneSide + 1)]
    centersSortedByHeight.sort(key = lambda x : x[0][0])
    for heightCluster in centersSortedByHeight : heightCluster.sort(key = lambda x : x[1][1])
    positionArray = [[0 for _ in range(2 * numberOfColumnsOnOneSide + 1)] for _ in range(len(centersSortedByHeight[0]))]
    for x in range(len(centersSortedByHeight)) :
        heightCluster = centersSortedByHeight[x]
        for y in range(len(heightCluster)) :
            point = heightCluster[y][1]
            positionArray[y][heightCluster[y][0]] = point
    return [positionArray, centersSortedByHeight]

def drawFigures(allCenters, centerColumnX) : 
    img = cv2.imread(PROSSESED_DIRNAME + "rotated.png")
    for point in allCenters:
        cX, cY = point[1][0], point[1][1]
        startPoint = (cX - RECTANGLE_SIZE, cY - RECTANGLE_SIZE)
        endPoint   = (cX + RECTANGLE_SIZE, cY + (RECTANGLE_SIZE * 2))
        if(cX > centerColumnX + 20) :
            startPoint = (startPoint[0], startPoint[1] - RECTANGLE_SIZE)
            endPoint   = (endPoint[0]  , endPoint[1]   - RECTANGLE_SIZE)
        color = (0,255,0) if(point[2] == CALCULATED) else (255,0,0)
        cv2.rectangle(img, startPoint, endPoint, color)
        if(SHOW_LABELS) : cv2.putText(img, str(point[0]), point[1], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.circle(img, point[1], 3, color, thickness=3)
    return img

# endregion CustomFunctions

def getPositionArray(img):
    print("Finding holes positions ...")
    IMG_WIDTH, IMG_HEIGHT, _ = img.shape
    cv2.imwrite(PROSSESED_DIRNAME + "rotated.png", img)

    # On récupère le centre de chaque rectangles formé autour des contours détectés par OpenCV
    centers = getContourCenters(img, IMG_WIDTH)

    # On regroupe chaque centre dans une colonne
    verticalClusters = getVerticalClusters(centers)
    numberOfColumnsOnOneSide = int(len(verticalClusters) / 2)

    # Marquage de la colonne du centre
    centerColumnX = int(np.mean(list(map(lambda a : a[0], verticalClusters[numberOfColumnsOnOneSide]))))
    if(DEBUG) : cv2.line(img, (centerColumnX, 0), (centerColumnX, 2040), (255,255,0), thickness= 3)

    # On regroupe chaque centre dans une ligne
    horizontalClusters, centerColumn = getHorizontalClusters(centers, numberOfColumnsOnOneSide, centerColumnX, centers, IMG_WIDTH, IMG_HEIGHT)

    # On calcules quelles abscisses correspondent à quelle colonne
    XposOfColumns = getXposOfColumns(verticalClusters)

    # On associe chaque centre de chaque ligne à sa colonne
    horizontalClustersWithX = associateHorizontalClusterWithXpos(horizontalClusters, XposOfColumns)

    # On peut alors identifier quelles lignes sont incomplètes et lesquelles le sont
    leftIncompleteLines, leftCompleteLines, rightIncompleteLines, rightCompleteLines = filterHorizontalClusters(horizontalClustersWithX, numberOfColumnsOnOneSide, centerColumnX)
    lineClusters = [leftIncompleteLines, rightIncompleteLines, leftCompleteLines, rightCompleteLines]

    # On calcule l'écart moyen entre chaque centre pour l'appliquer aux centres manquants
    leftxOffset, leftyOffset, rightxOffset, rightyOffset = getOffsets(leftCompleteLines, rightCompleteLines)

    # On complète les centres que nous n'avions pas pu détecter
    leftCompletedLines, rightCompletedLines = completeHorizontalClusters(leftIncompleteLines, rightIncompleteLines, leftxOffset, leftyOffset, rightxOffset, rightyOffset, numberOfColumnsOnOneSide)

    # On regroupe puis applati les points pour pouvoir les utiliser plus simplement
    leftLines = leftCompletedLines + leftCompleteLines
    rightLines = rightCompletedLines + rightCompleteLines
    allCenters = [x for list in (leftLines + rightLines + [centerColumn]) for x in list]

    # On complète les centres manquants sur la colonne du milieu
    allCenters, w, centersSortedByHeight, yOffset = fillMissingCenterColumn(allCenters, numberOfColumnsOnOneSide, centerColumn, verticalClusters)

    # On complète les lignes manquantes vers le bas
    allCenters = fillMissingHorizontalClusters(centersSortedByHeight, allCenters, numberOfColumnsOnOneSide, yOffset, w)
    allCenters = removeDuplicates(allCenters, 50)

    # On trace un rectangle et un chiffre par point
    if(DEBUG) : img = drawFigures(allCenters, centerColumnX)

    # On remplit le tableau des positions de chaque puit par rapport à la grille
    positionArray, centersSortedByHeight = populateArray(allCenters, numberOfColumnsOnOneSide)

    if(DEBUG) : cv2.imwrite(PROSSESED_DIRNAME + "recognition.png", img)

    printPositionArray(positionArray)

    return [positionArray, centerColumnX]

if __name__ == "__main__" :
    imgPath = "Data/Brightfield/BF no cells.jpg"
    angle = getBestAngle(imgPath)
    img = load_image(imgPath, angle)
    positionArray, centerColumnX = getPositionArray(img)
    printPositionArray(positionArray)
