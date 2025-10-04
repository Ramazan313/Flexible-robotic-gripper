import cv2
import Hesaplama_Real_time_Filtreleme

###################################
webcam = False
path = 'son.jpg'
cap = cv2.VideoCapture(1)
#cap.set(10,100)
#cap.set(3,480)
#cap.set(4,640)
scale = 3    # dönüştürme katsayısı
wP = int(210*scale)
hP= int(297*scale)
###################################

while True:
    if webcam:
        success, img = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            continue  # döngünün bu adımını atlar
    else:
        img = cv2.imread(path)
        #print(img.shape)

    imgContours , conts = Hesaplama_Real_time_Filtreleme.getContours(img,minArea=3500,filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = Hesaplama_Real_time_Filtreleme.warpImg(img, biggest, wP,hP)
        _, conts2 = Hesaplama_Real_time_Filtreleme.getContours(imgWarp,
                                                               minArea=1000, filter=4,
                                                               cThr=[20, 20], draw=False)
        imgContours2 = imgWarp.copy()

        if len(conts2) != 0:
            for obj in conts2:
                if len(obj[2]) == 4:  # Sadece 4 köşeli olanlarla ölçüm yap
                    cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                    nPoints = Hesaplama_Real_time_Filtreleme.reorder(obj[2])
                    nW = round((Hesaplama_Real_time_Filtreleme.findDis(nPoints[0][0], nPoints[1][0]) / scale) / 10, 1)
                    nH = round((Hesaplama_Real_time_Filtreleme.findDis(nPoints[0][0], nPoints[2][0]) / scale) / 10, 1)
                    print(f"En: {nW} cm, Boy: {nH} cm")
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                    (nPoints[1][0][0], nPoints[1][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                    (nPoints[2][0][0], nPoints[2][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)
                    x, y, w, h = obj[3]
                    cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)
                    cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)

        cv2.imshow('measurement', imgContours2)

    img = cv2.resize(img,(0,0),None,1,1)
    cv2.imshow('Original',img)
    cv2.waitKey(1)