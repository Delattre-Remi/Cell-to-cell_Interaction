### Les rectangles jaunes signifient qu'un contours à été supprimé sur la base de son rectangle le plus petit
import cv2
import numpy as np
from utils import rotate_image

def discriminateContour(cnt, img):
    area = cv2.contourArea(cnt)
    if(area < 450 or area > 800) :
        # Show which contours were close to being ruled out ::: if(area > 200 and area < 1000) : cv2.drawContours(img, cnt, -1, (255,0,0), 5)
        return False
    x,y,w,h = cv2.boundingRect(cnt)
    ratio = w / h if w > h else h / w
    if(ratio < 2) : return True
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255, 255), 10)
    return False

def getBestAngle(imgPath):
    done = False
    angle = 1.25
    finalDistances = []
    minDist = 9999

    print("Finding best angle ...")
    while not done :
        img = cv2.imread(imgPath)
        img = rotate_image(img, angle)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 127, 255, 0)

        # Trouver tous les contours trouvables sur l'image
        contours, _ = cv2.findContours(th, 2, 1)

        # Filter seulement les contours qui nous intéressent (ceux entre 500 et 800 pixels de surface)
        contours = list(filter(lambda a : discriminateContour(a, img) ,contours))

        # Coefficient pour la taille des rectangles
        size = 33
        centers = []
        distances = []

        # Créer un rectangle autour de chaque contour choisi
        for c in contours : 
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

        centers.sort()

        # Créer les colonnes de puits
        last = centers[0][0]
        cluster = []
        clusters = []
        for point in centers:
            if(np.abs(point[0] - last) < 50) : 
                cluster.append(point)
                distances.append(np.abs(point[0] - last))
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

        # Colonne du centre
        centerColumnX = int(np.mean(list(map(lambda a : a[0] ,clusters[int(len(verticalClustersLengths) / 2)]))))
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
        
        distSum = np.sum(distances)

        # Coloriage du contour selectionné
        cv2.drawContours(img, contours, -1, (0,255,0), 2)
        if(distSum < 85) : cv2.imwrite("AngleDetermination/Recognition #{} Angle {:.2f} ! Distance {:.0f}.png".format(int(angle * 100), angle, distSum), img)
        
        finalDistances.append([distSum, angle])
        if(distSum - minDist > 50) : done = True

        elif(distSum < minDist) : minDist = distSum

        angle += 0.01

    contour_areas = [cv2.contourArea(a) for a in contours]
    print("Mean area : {:.2f}\nMax area : {}\nMin area: {}".format(np.mean(contour_areas), np.amax(contour_areas), np.amin(contour_areas)))
    finalDistances = np.array(finalDistances)
    bestAngle = finalDistances[finalDistances[:,0].argsort()][0][1]
    print("Found best angle is : {:.2f}".format(bestAngle))

    return bestAngle
    
if __name__ == "__main__":
    bestAngle = getBestAngle("Data/Brightfield/BF no cells.jpg")
    