import cv2
import numpy as np

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def hasValueBetween(arr, value, thresh):
    bool = False
    for val in arr:
        if(val[0] >= value - thresh and val[0] <= value + thresh): 
            bool = True
            break
    return bool

done = False
angle = 0
negative = False
changingDir = False
finalDistances = []
minDist = 9999
while not done :
    img = cv2.imread("Data/Brightfield/BF no cells.jpg")
    img = rotate_image(img, angle * (-1))

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
        startPoint = (cX - size, cY - size)
        endPoint = (cX + size, cY + (size * 2))
        width, height = endPoint[0] - startPoint[0], endPoint[1] - startPoint[1]
        center = (int(startPoint[0] + width/2), int(startPoint[1] + height/2))
        cv2.circle(img, center, 2, (0,0,255), thickness=5)
        cv2.rectangle(img, startPoint, endPoint, (255, 0, 0))
        centers.append(center)

    centerClusteredByX = centers
    centerClusteredByX.sort()

    # Créer les colonnes de puits
    last = centerClusteredByX[0][0]
    cluster = []
    clusters = []
    distances = []
    for point in centerClusteredByX:
        distance = np.abs(point[0] - last)
        if(distance < 5) : 
            cluster.append(point)
            distances.append(distance)
        else:
            if(len(cluster) > 0) : clusters.append(cluster)
            cluster = []
        last = point[0]

    
    listPointsInRow = []
    for cluster in clusters:
        pointsInRow = []
        for point in cluster:
            pointsInRow = list(filter(lambda a : np.abs(a[0] - point[0]) < 20 ,cluster))
            listPointsInRow.append(len(pointsInRow))

    for cluster in clusters:
        cv2.line(img, cluster[0], cluster[-1], (0,0,0), thickness=3)
    
    distSum = np.sum(distances)

    final = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imwrite("Recogs/minus/recognition{}!{:.2f}.png".format(angle, distSum), img)
    
    finalDistances.append([distSum, angle])
    if(distSum - minDist > 50) :
        done = True
    elif(distSum < minDist) :
        minDist = distSum
        print(minDist)

    angle += 0.01

print(finalDistances)