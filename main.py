import os
import cv2
import numpy as np
from utils.bundleAjust import bundleAdjustment
from utils.dense import denseMatch, denseReconstruction, outputPly
from utils.fundamental import default, RansacImplementation
from utils.getPose import getPose
from utils.graph import createGraph, triangulateGraph, showGraph, visualizeDense
from utils.mergeGraph import mergeG, removeOutlierPts
from utils.paresDescript import getPairSIFT




def mergeAllGraph(gL, imsize):
    graphMerged = gL[0]
    # merge partial views
    for i in range(len(gL) - 1):
        graphMerged = updateMerged(graphMerged, gL[i+1], imageSize)
    return graphMerged

def updateMerged(gA, gB, imsize):
    gt = mergeG(gA, gB)
    gt = triangulateGraph(gt, imsize)
    gt = bundleAdjustment(gt, False)
    gt = removeOutlierPts(gt, 10)
    gt = bundleAdjustment(gt)
    return gt

if __name__ == "__main__":

    #---------------------------SET PARAMETERS
    maxSize = 640 #max resolution
    folderImages = "D:\sensor fusion\SFM python\SFM python\example\Valbonne_Church"
    debug = False
    outName = "sculpture" #out name for ply file (open with mesh lab to see poitn cloud)
    validFile = ['jpg','png','JPG','ppm', 'pgm'] # valid type of images
    # TODO focal length calculation
    # This value should work with imagenes 480x640 focalLength 4mm (no guaranties)
    f = 719.5459
    #f = 2360

    # ---------------------------SET PARAMETERS


    algoFundamentalMatrix = RansacImplementation

    graphList = []

    #load imagenes
    listFiles = os.listdir(folderImages)
    listImages = list(filter(lambda x : x.split('.')[-1] in validFile, listFiles ))




    #load images
    listFrames = list(map(lambda x : cv2.imread(folderImages+x), listImages))

    imageSize = listFrames[0].shape
    print ("Original Dimensions ",imageSize)
    #scale images based on maxSize
    if imageSize[0] > maxSize:
        print ("Scaler")
        print ("Size image ",imageSize," max size ",maxSize)
        #480 640 works
        listFrames = list(map(lambda x: np.transpose(cv2.resize(x,(640,480)),axes=[1,0,2]), listFrames))
        imageSize = listFrames[0].shape
        print ("Result size ",imageSize)

    #matrix K
    K = np.eye(3)
    K[0][0] = f
    K[1][1] = f

    graphList = [0 for i in range(len(listFrames)-1)]
    # get feature descriptors using SIFT. We couls use other local descriptors such as SURF
    #Sequence of images
    print ("Computing SIFT")
    for i in range(len(listFrames)-1):
        keypointsA, keypointsB = getPairSIFT(listFrames[i], listFrames[i+1], show=debug)


        #Compute Fundamental Matrix and Essential Matrix
        #TODO get more transformations
        if type(keypointsA[0]) == np.ndarray:
            assert(len(keypointsA.shape) == 2)
            assert (len(keypointsB.shape) == 2)
            pointsA = keypointsA
            pointsB = keypointsB
        else:
            pointsA = np.array([(keypointsA[idx].pt) for idx in range(len(keypointsA))]).reshape(-1, 1, 2)
            pointsB = np.array([(keypointsB[idx].pt) for idx in range(len(keypointsB))]).reshape(-1, 1, 2)
        pointsA = pointsA[:,[1,0]]
        pointsB = pointsB[:, [1, 0]]

        F = np.array(algoFundamentalMatrix(pointsA, pointsB))
        Fmat = F[0]
        K = np.array(K)
        E = np.dot(np.transpose(K), np.dot(Fmat, K))

        # Get camara pose
        Rtbest = getPose(E, K, np.hstack([pointsA, pointsB]), imageSize)

        #plot
        graphList[i] = createGraph(i, i+1, K, pointsA, pointsB, Rtbest, f)

        #Triangulation
        graphList[i] = triangulateGraph(graphList[i], imageSize)

        #visualization
        #showGraph(graphList[i], imageSize)

        #Bundle ajustement
        graphList[i]=bundleAdjustment(graphList[i])

        #Visualization after optimization
        # showGraph(graphList[i], imageSize)

    gM = mergeAllGraph(graphList, imageSize)
    print ("Merge final graphs")
    #Partial map visualization
    showGraph(gM,imageSize)
    #Dense matching
    for i in range(len(listFrames)-1):
        graphList[i] = denseMatch(graphList[i],listFrames[i],
                                  listFrames[i+1], imageSize, imageSize)

    print ("Final Dense match")
    print ("Dense Triangulation Initialization")
    #Dense reconstruction
    for i in range(len(listFrames) - 1):
        graphList[i] = denseReconstruction(graphList[i], gM, K, imageSize)
    print ("Dense reconstruct")
    data = visualizeDense(graphList, gM, imageSize)

    outputPly(data,outName)












