import cv2
import copy
from utils import PROSSESED_DIRNAME, DEBUG, RECTANGLE_SIZE, load_image
from holePositionDetermination import getPositionArray
from angleDetermination import getBestAngle

def getContourCenters(img, isTcell = False):
    if(DEBUG) : 
        if(isTcell) : test_img = cv2.imread(PROSSESED_DIRNAME + "tCellsLast.png")
        else : test_img = cv2.imread(PROSSESED_DIRNAME + "leukemicCellsLast.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray, 127, 255, 0)

    # Trouver tous les contours trouvables sur l'image
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter seulement les contours qui nous intéressent
    contours = list(filter(lambda a : cv2.contourArea(a) < 2000 and cv2.contourArea(a) > 10 ,contours))

    centers = []
    for c in contours : 
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if(DEBUG) : 
            cv2.drawContours(test_img, contours, -1, (0,255,0), 3)
            cv2.circle(test_img, (cX, cY), 5, (255,0,0), thickness=3)
        centers.append((cX,cY))

    if(DEBUG) : 
        if(not isTcell) : cv2.imwrite(PROSSESED_DIRNAME + "Step_5_leukemic_cells_center_determination.png", test_img)
        else : cv2.imwrite(PROSSESED_DIRNAME + "Step_7_t_cells_center_determination.png", test_img)
    centers.sort()
    return centers

def rectangleFromPoint(point, RECTANGLE_SIZE, centerColumnX):
    cX, cY = point[0], point[1]
    startPoint = (cX - RECTANGLE_SIZE, cY - RECTANGLE_SIZE)
    endPoint   = (cX + RECTANGLE_SIZE, cY + (RECTANGLE_SIZE * 2))
    if(cX > centerColumnX + 20) :
        startPoint = (startPoint[0], startPoint[1] - RECTANGLE_SIZE)
        endPoint   = (endPoint[0]  , endPoint[1]   - RECTANGLE_SIZE)
    return (startPoint, endPoint)

def isWithinRectangle(point, rectCenter, RECTANGLE_SIZE, centerColumnX):
    rectStart, rectEnd = rectangleFromPoint(rectCenter, RECTANGLE_SIZE, centerColumnX)
    cX, cY = point[0], point[1]
    if(cX > rectStart[0] and cX < rectEnd[0] and cY > rectStart[1] and cY < rectEnd[1]):
        return (rectStart, rectEnd)
    return None

def assignCenterToPositionArray(centers, positionArray, RECTANGLE_SIZE, centerColumnX, isTcell = False):
    if(DEBUG) : 
        if(isTcell) : test_img = cv2.imread(PROSSESED_DIRNAME + "rotated.png")
        else : test_img = cv2.imread(PROSSESED_DIRNAME + "rotated.png")
        passes = 0
    positionArrayLeft = copy.deepcopy(positionArray)
    foundPositions = []
    for c in centers:
        found = False
        for line in positionArrayLeft:
            tmpLine = []
            for arrayPos in line:
                rect = isWithinRectangle(c, arrayPos, RECTANGLE_SIZE, centerColumnX)
                if(DEBUG) : passes += 1
                if(not rect is None) :
                    tmpLine.append(arrayPos)
                    try : line.remove(arrayPos)
                    except : pass
                    found = True
                    if(DEBUG) :
                        cv2.rectangle(test_img, rect[0], rect[1], (255,0,0))
                        cv2.circle(test_img, arrayPos, 3, (255,0,0), 5)
                    break
            if(len(tmpLine) > 0) : foundPositions.append(tmpLine)
            if(found) : break

    if(DEBUG) : 
        if(not isTcell) : cv2.imwrite(PROSSESED_DIRNAME + "Step_6_cell_positionning.png", test_img)
        else : cv2.imwrite(PROSSESED_DIRNAME + "Step_8_cell_superpositioning.png", test_img)

    return foundPositions    

def getInteractionArray(BF_imgPath, leukemicCellsImgPath, tCellsImgPath):
    angle = getBestAngle(BF_imgPath)

    BF_img = load_image(BF_imgPath, angle)
    positionArray, centerColumnX = getPositionArray(BF_img)

    print("Finding interaction zones ...")

    leukemicCellsImg = load_image(leukemicCellsImgPath, angle)
    leukemicCellsImg = cv2.resize(leukemicCellsImg, BF_img.shape[:2], interpolation = cv2.INTER_AREA)
    if(DEBUG) : cv2.imwrite(PROSSESED_DIRNAME + "leukemicCellsLast.png", leukemicCellsImg)

    tCellsImg = load_image(tCellsImgPath, angle)
    tCellsImg = cv2.resize(tCellsImg, BF_img.shape[:2], interpolation = cv2.INTER_AREA)
    if(DEBUG) : cv2.imwrite(PROSSESED_DIRNAME + "tCellsLast.png", tCellsImg)

    leukemicCellsCenters = getContourCenters(leukemicCellsImg)
    tCellsCenters = getContourCenters(tCellsImg, isTcell = True)
    positionsWhereLeukemicCellsAre = assignCenterToPositionArray(leukemicCellsCenters, positionArray, RECTANGLE_SIZE, centerColumnX)
    interactionPositions = assignCenterToPositionArray(tCellsCenters, positionsWhereLeukemicCellsAre, RECTANGLE_SIZE, centerColumnX, isTcell=True)
    interactionPositions = [x[0] for x in interactionPositions]
    print(interactionPositions)

    numberOfHoles = len(positionArray[0]) * len(positionArray)
    numberOfValidLeukemicCells = len(leukemicCellsCenters)
    numberOfValidTCells = len(tCellsCenters)
    numberOfinteractions = len(interactionPositions)
    pourcentageOfValidLeukemicCells = (numberOfValidLeukemicCells / numberOfHoles) * 100 
    pourcentageOfValidTCells = (numberOfValidTCells / numberOfHoles) * 100 
    pourcentageOfInteractions = (numberOfinteractions / numberOfHoles) * 100 

    print("\033[32mThere is a total of {} holes to be filled, which {} ({:.2f}%) were filled with Leukemic cells and {} ({:.2f}%) were filled with T cells which results in {} ({:.2f}%) interaction zones\033[0m".format(
        numberOfHoles, numberOfValidLeukemicCells, pourcentageOfValidLeukemicCells, numberOfValidTCells, pourcentageOfValidTCells, numberOfinteractions, pourcentageOfInteractions))

    return [interactionPositions, numberOfHoles]

if __name__ == "__main__" :
    BF_imgPath = "Data/Brightfield/BF no cells.jpg"
    leukemicCellsImgPath = "Data/Leukemic cells/500.png"
    tCellsImgPath = "Data/T-cells/500.png"
    getInteractionArray(BF_imgPath, leukemicCellsImgPath, tCellsImgPath)