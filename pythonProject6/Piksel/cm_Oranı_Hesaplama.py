import cv2
import numpy as np


def getContours(img, cThr=[100, 100], showCanny=True, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gri',imgGray)
    #alpha = 0.8  # kontrast oranı (1.0 = değişiklik yok)
   # beta = -50  # parlaklık kaydırması (- değerler koyulaştırır)
    #imgGray = cv2.convertScaleAbs(imgGray, alpha=alpha, beta=beta)
   # cv2.imshow('tonlu',imgGray)

    # Gölgeyi azaltmak için Adaptive Threshold
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
 ###   cv2.imshow('GaussianBlur',imgBlur)
    #cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
# src:gray formatında bir frame, maxfalue:eşikleme sonrası atanacak max değer, cv2.ADAPTIVE_THRESH_MEAN_C komşu piksellerin ortalamasını alır.
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C komluulara gaussian ağırlığı verir ve ona göre hesaplar.
#cv2.THRESH_BINARY normal siyah beyaz yapar. cv2.THRESH_BINARY_INV siyah-beyazı ters çevirir.
# blockSize: her piksel için bakılacak komşuluk boyutu 11x11 13x13 gibi(3,5,7,9,11..)
# C: küçük c  daha fazla beyaz çıkar, büyük c de ise daha fazla siyah çıkar.
    imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 27, 0.5)
 ###   cv2.imshow('adaptiveTreshold',imgThresh)
    # Küçük gürültüleri temizleme
    kernel = np.ones((3, 3), np.uint8)
    imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)
    imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)
 ###   cv2.imshow('morpologyEx',imgThresh)
    imgCanny = cv2.Canny(imgThresh, cThr[0], cThr[1])
 ###   cv2.imshow('canny',imgCanny)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)

    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalContours


def reorder(myPoints):
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def warpImg(img, points, w, h, pad=0):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    return imgWarp


def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5



