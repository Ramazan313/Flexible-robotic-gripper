import cv2
import cm_Oranı_Hesaplama as cm
import time

###################################
webcam = True
path = 'beyazLevha.jpg'  
cap = cv2.VideoCapture(1)
scale = 26/10  # değeri arttıkça uzunluk azalır
wP = int(210*scale)
hP = int(286*scale)
###################################

while True:
    if webcam:
        success, img = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            continue
    else:
        img = cv2.imread(path)
        time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("yeni11.jpg", img)
        print("Fotoğraf çekildi.")

    imgContours, conts = cm.getContours(img, minArea=50000, filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        imgWarp = cm.warpImg(img, biggest, wP, hP)

        imgContours2, conts2 = cm.getContours(imgWarp,
                                               minArea=2000, filter=4,
                                               cThr=[50,50], draw=False)
        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2, [obj[2]], True, (0,255,0), 2)
                nPoints = cm.reorder(obj[2])
                nW = round((cm.findDis(nPoints[0][0], nPoints[1][0]) / scale) / 10, 1)
                nH = round((cm.findDis(nPoints[0][0], nPoints[2][0]) / scale) / 10, 1)

                x, y, w, h = obj[3]
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow('Measurement', imgContours2)

    img = cv2.resize(img, (0, 0), None, 1, 1)
    cv2.imshow('Original', img)
    cv2.waitKey(1)


