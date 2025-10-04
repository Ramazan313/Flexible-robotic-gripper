import cv2
import numpy as np

def reorder_pts(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def find_a4_contour(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    edges = cv2.Canny(blur, 50, 150)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4 and cv2.contourArea(approx)>30000:
            return approx
    return None

def detect_inner_sheet(warp,
                       lab_thresh=0.7,
                       hsv_thresh=0.7,
                       presence_thresh=0.08):
    h, w = warp.shape[:2]

    # --- Lab kanallarıyla beyaz oranı
    lab = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB)
    A = lab[:,:,1]; B = lab[:,:,2]
    lab_white = cv2.inRange(A, 128-5, 128+5) & cv2.inRange(B, 128-5, 128+5)
    lab_ratio = np.count_nonzero(lab_white) / lab_white.size

    # --- HSV ile beyaz oranı
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]; V = hsv[:,:,2]
    hsv_white = (S < 30) & (V > 200)
    hsv_ratio = np.count_nonzero(hsv_white) / hsv_white.size

    # Eğer hem Lab hem HSV beyaz oranı yüksekse: gerçek beyaz levha
    if lab_ratio > lab_thresh and hsv_ratio > hsv_thresh:
        # mask görselleştirme için combine edelim
        white_mask = (lab_white & hsv_white).astype(np.uint8)*255
        return "white_sheet", lab_ratio, hsv_ratio, white_mask

    # --- Renkli nesne var mı diye adaptiveThreshold kontur analizi
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cnts,_ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_sum = sum(cv2.contourArea(c) for c in cnts if cv2.contourArea(c)>500)
    presence_ratio = area_sum / (w*h)
    if presence_ratio > presence_thresh:
        return "colored_sheet", lab_ratio, presence_ratio, clean

    return "no_sheet", lab_ratio, presence_ratio, clean

def main():
    cap = cv2.VideoCapture(1)
    W, H = 600, 800

    while True:
        ret, frame = cap.read()
        if not ret: break
        orig = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        a4 = find_a4_contour(gray)
        if a4 is not None:
            cv2.drawContours(frame, [a4], -1, (0,255,0), 2)

            pts = reorder_pts(a4)
            dst = np.array([[0,0],[W,0],[W,H],[0,H]],dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            warp = cv2.warpPerspective(orig, M, (W,H))

            status, r1, r2, mask = detect_inner_sheet(
                warp,
                lab_thresh=0.85,
                hsv_thresh=0.85,
                presence_thresh=0.03
            )

            # debug pencereleri
            cv2.imshow("Warped A4", warp)
            cv2.imshow("Mask", mask)

            if status=="white_sheet":
                txt = f"Beyaz levha (Lab:{r1:.2f},HSV:{r2:.2f})"
                col = (0,255,0)
            elif status=="colored_sheet":
                txt = f"Renkli levha (obj:{r2:.2f})"
                col = (0,0,255)
            else:
                txt = "Levhâ yok"
                col = (0,255,255)

            cv2.putText(frame, txt, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        else:
            cv2.putText(frame, "A4 bulunamadi", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Sonuc", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
