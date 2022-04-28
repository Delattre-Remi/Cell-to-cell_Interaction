import cv2
import copy
from utils import PROSSESED_DIRNAME, DEBUG, RECTANGLE_SIZE, load_image
from holePositionDetermination import getPositionArray
from angleDetermination import getBestAngle
import time

# Developped by https://github.com/Mostuniqu3/

def getContourCenters(img, superposedImage = [], RENDER_STEP578 = False, RENDER_SUPERPOS = False, isTcell = False):
    if(RENDER_STEP578):
        if(isTcell) : test_img = cv2.imread(PROSSESED_DIRNAME + "tCellsLast.png")
        else        : test_img = cv2.imread(PROSSESED_DIRNAME + "leukemicCellsLast.png")
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
        if(RENDER_SUPERPOS) : cv2.drawContours(superposedImage, contours, -1, (0,0,255) if isTcell else (255,0,0), 5)
        if(RENDER_STEP578): 
            cv2.drawContours(test_img, contours, -1, (0,255,0), 3)
            cv2.circle(test_img, (cX, cY), 5, (255,0,0), thickness=3)
        centers.append((cX,cY))


    if(RENDER_STEP578): 
        if(not isTcell) : cv2.imwrite(PROSSESED_DIRNAME + "Step_5_leukemic_cells_center_determination.png", test_img)
        else            : cv2.imwrite(PROSSESED_DIRNAME + "Step_7_t_cells_center_determination.png", test_img)
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
    if(cX > rectStart[0] and cX < rectEnd[0] and cY > rectStart[1] and cY < rectEnd[1]): return (rectStart, rectEnd)
    return None

def assignCenterToPositionArray(centers, positionArray, RECTANGLE_SIZE, centerColumnX, superposedImage = [], RENDER_STEP578 = False, RENDER_SUPERPOS = False, isTcell = False):
    if(DEBUG and not isTcell) : test_img = cv2.imread(PROSSESED_DIRNAME + "rotated.png")
    if(RENDER_STEP578 and isTcell) : test_img = cv2.imread(PROSSESED_DIRNAME + "rotated.png")
    positionArrayLeft = copy.deepcopy(positionArray)
    foundPositions = []
    for c in centers:
        found = False
        for line in positionArrayLeft:
            tmpLine = []
            for arrayPos in line:
                rect = isWithinRectangle(c, arrayPos, RECTANGLE_SIZE, centerColumnX)
                if(not rect is None) :
                    tmpLine.append(arrayPos)
                    try : line.remove(arrayPos)
                    except : pass
                    found = True
                    if(RENDER_STEP578 and (isTcell or DEBUG)) :
                        cv2.rectangle(test_img, rect[0], rect[1], (255,0,0))
                        if(RENDER_SUPERPOS) : cv2.rectangle(superposedImage, rect[0], rect[1], (0,0,0))
                        cv2.circle(test_img, arrayPos, 3, (255,0,0), 5)
                    break
            if(len(tmpLine) > 0) : foundPositions.append(tmpLine)
            if(found) : break

    if(DEBUG and not isTcell) : cv2.imwrite(PROSSESED_DIRNAME + "Step_6_cell_positionning.png", test_img)
    if(RENDER_STEP578 and isTcell) : cv2.imwrite(PROSSESED_DIRNAME + "Step_8_cell_superpositioning.png", test_img)

    return foundPositions    

class Analyser:
    def __init__(self, BF_imgPath):
        self.angle = getBestAngle(BF_imgPath)
        self.BF_img = load_image(BF_imgPath, self.angle)
        self.superpositionImage = load_image(BF_imgPath, self.angle)
        self.positionArray, self.centerColumnX = getPositionArray(self.BF_img)
        self.RENDER_SUPERPOS = False
        self.RENDER_STEP578 = False

    def setRenderSuperPos(self, bool):
        self.RENDER_SUPERPOS = bool

    def setRenderSteps(self, bool):
        self.RENDER_STEP578 = bool

    def getInteractionArray(self, leukemicCellsImgPath, tCellsImgPath):
        start = time.time()
        print("Finding interaction zones ...")

        leukemicCellsImg = load_image(leukemicCellsImgPath, self.angle)
        leukemicCellsImg = cv2.resize(leukemicCellsImg, self.BF_img.shape[:2], interpolation = cv2.INTER_AREA)
        cv2.imwrite(PROSSESED_DIRNAME + "leukemicCellsLast.png", leukemicCellsImg)

        tCellsImg = load_image(tCellsImgPath, self.angle)
        tCellsImg = cv2.resize(tCellsImg, self.BF_img.shape[:2], interpolation = cv2.INTER_AREA)
        cv2.imwrite(PROSSESED_DIRNAME + "tCellsLast.png", tCellsImg)

        leukemicCellsCenters = getContourCenters(leukemicCellsImg, self.superpositionImage, self.RENDER_STEP578, self.RENDER_SUPERPOS)
        tCellsCenters = getContourCenters(tCellsImg, self.superpositionImage, self.RENDER_STEP578, self.RENDER_SUPERPOS, isTcell = True)
        positionsWhereLeukemicCellsAre = assignCenterToPositionArray(leukemicCellsCenters, self.positionArray, RECTANGLE_SIZE, self.centerColumnX, self.RENDER_STEP578, self.RENDER_SUPERPOS)
        interactionPositions = assignCenterToPositionArray(tCellsCenters, positionsWhereLeukemicCellsAre, RECTANGLE_SIZE, self.centerColumnX, self.superpositionImage, self.RENDER_STEP578, self.RENDER_SUPERPOS, isTcell=True)
        interactionPositions = [x[0] for x in interactionPositions]
        stats = {
            "numberOfHoles" : len(self.positionArray[0]) * len(self.positionArray),
            "numberOfValidLeukemicCells" : len(leukemicCellsCenters),
            "numberOfValidTCells" : len(tCellsCenters),
            "numberOfinteractions" : len(interactionPositions)
        }

        stats["pourcentageOfValidLeukemicCells"] = (stats["numberOfValidLeukemicCells"] / stats["numberOfHoles"]) * 100
        stats["pourcentageOfValidTCells"] = (stats["numberOfValidTCells"] / stats["numberOfHoles"]) * 100
        stats["pourcentageOfInteractions"] = (stats["numberOfinteractions"] / stats["numberOfHoles"]) * 100 

        if(self.RENDER_SUPERPOS) : cv2.imwrite(PROSSESED_DIRNAME + "superposed.png", self.superpositionImage)
        print('Took {:.2f} ms'.format(float(time.time() - start) * 1000))
        return [interactionPositions, stats]

if __name__ == "__main__" :
    BF_imgPath = "Data/Brightfield/BF no cells.jpg"
    leukemicCellsImgPath = "Data/Leukemic cells/500.png"
    tCellsImgPath = "Data/T-cells/500.png"
    an = Analyser(BF_imgPath)
    an.getInteractionArray(leukemicCellsImgPath, tCellsImgPath)