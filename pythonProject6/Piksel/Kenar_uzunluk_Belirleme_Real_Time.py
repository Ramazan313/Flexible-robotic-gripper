import cv2
import Hesaplama_Real_time

###################################
webcam = True
path = 'son.jpg'
cap = cv2.VideoCapture(1)
#cap.set(10,100)
#cap.set(3,480)
#cap.set(4,640)
scale = 28/10      # dönüştürme katsayısı
wP = int(420)
hP= int(594)
###################################

while True:
    if webcam:
        success, img = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            continue  # döngünün bu adımını atlar
    else:
        img = cv2.imread(path)
        print(img.shape)

    imgContours , conts = Hesaplama_Real_time.getContours(img,minArea=50000,filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = Hesaplama_Real_time.warpImg(img, biggest, wP,hP)
        imgContours2, conts2 = Hesaplama_Real_time.getContours(imgWarp,
                                                 minArea=2000, filter=4,
                                                 cThr=[50,50],draw = False)

        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                nPoints = Hesaplama_Real_time.reorder(obj[2])
                nW = round((Hesaplama_Real_time.findDis(nPoints[0][0], nPoints[1][0]) / scale) / 10, 1)
                nH = round((Hesaplama_Real_time.findDis(nPoints[0][0], nPoints[2][0]) / scale) / 10, 1)
                print(nW)
                print(nH)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        cv2.imshow('measurement', imgContours2)

    img = cv2.resize(img,(0,0),None,1,1)
    cv2.imshow('Original',img)
    cv2.waitKey(1)