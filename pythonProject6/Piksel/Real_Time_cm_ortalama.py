import cv2
import Hesaplama_Real_time
##
###################################
webcam = True
path = 'A4.jpg'
cap = cv2.VideoCapture(1)
# cap.set(10,100)
# cap.set(3,480)
# cap.set(4,640)
scale = 28 / 10  # dönüştürme katsayısı
wP = int(150 * scale)
hP = int(210 * scale)
###################################

# Son 10 değeri saklamak için listeler
nW_list = []
nH_list = []

while True:
    if webcam:
        success, img = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            continue  # döngünün bu adımını atlar
    else:
        img = cv2.imread(path)
        print(img.shape)

    imgContours, conts = Hesaplama_Real_time.getContours(img, minArea=12000, filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        # print(biggest)
        imgWarp = Hesaplama_Real_time.warpImg(img, biggest, wP, hP)
        imgContours2, conts2 = Hesaplama_Real_time.getContours(imgWarp,
                                                               minArea=2000, filter=4,
                                                               cThr=[50, 50], draw=False)

        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = Hesaplama_Real_time.reorder(obj[2])
                nW = round((Hesaplama_Real_time.findDis(nPoints[0][0], nPoints[1][0]) / scale) / 10, 1)
                nH = round((Hesaplama_Real_time.findDis(nPoints[0][0], nPoints[2][0]) / scale) / 10, 1)

                # Listeye yeni değerler ekleniyor
                nW_list.append(nW)
                nH_list.append(nH)

                # Liste 10 değeri aştığında, en eski değeri çıkar
                if len(nW_list) > 10:
                    nW_list.pop(0)
                if len(nH_list) > 10:
                    nH_list.pop(0)

                # Son 10 değerin ortalamasını hesapla
                avg_nW = round(sum(nW_list) / len(nW_list), 1) if len(nW_list) > 0 else 0
                avg_nH = round(sum(nH_list) / len(nH_list), 1) if len(nH_list) > 0 else 0

                print(f"Ortalama nW: {avg_nW} cm")
                print(f"Ortalama nH: {avg_nH} cm")

                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(avg_nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(avg_nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1.5,
                            (255, 0, 255), 2)
        cv2.imshow('measurement', imgContours2)

    img = cv2.resize(img, (0, 0), None, 1, 1)
    cv2.imshow('Original', img)
    cv2.waitKey(1)
