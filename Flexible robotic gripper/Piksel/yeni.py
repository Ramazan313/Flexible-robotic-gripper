import cv2
import numpy as np

def detect_a4_and_check_inner_area(img):
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Görüntüyü bulanıklaştır
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Canny kenar tespiti
    edges = cv2.Canny(blurred, 50, 150)

    # Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # A4 kağıdının dikdörtgen şekli için en büyük konturu bul
    a4_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:  # A4 kağıdının dört kenarı olmalı
            a4_contour = approx
            break

    if a4_contour is not None:
        # A4 kağıdını işaretle
        cv2.drawContours(img, [a4_contour], -1, (0, 255, 0), 3)

        # A4 kağıdının iç kısmını maskele
        mask = np.zeros_like(img[:, :, 0])  # Sadece gri tonlama maskesi
        cv2.fillPoly(mask, [a4_contour], 255)  # A4 kağıdını maskele
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # Maskelenmiş bölgede beyaz olmayan renkleri tespit et
        non_white_area = masked_img[np.where((masked_img[:, :, 0] != 255) &
                                              (masked_img[:, :, 1] != 255) &
                                              (masked_img[:, :, 2] != 255))]

        # Eğer beyaz olmayan bir renk varsa bildir
        if non_white_area.size > 0:
            cv2.putText(img, "Farkli renkli levha bulundu!", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(img, "Sadece beyaz alan var.", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(img, "A4 kağıdı bulunamadı!", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return img

# Test için bir görüntü açalım (dosyadan veya web kameradan)
cap = cv2.VideoCapture(1)  # Web kamerayı açıyoruz
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_a4_and_check_inner_area(frame)

    # Sonucu ekranda göster
    cv2.imshow("Sonuc", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
