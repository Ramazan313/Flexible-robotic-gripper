import serial
import serial.tools.list_ports
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import cv2
from PIL import Image, ImageTk
import threading
import cm_Oranı_Hesaplama as cm

CH340_VID = '1A86'
CH340_PID = '7523'
MEGA_VID = '2341'
MEGA_PID = '0042'

ser = None
cap = None
frame = None
already_sent = False

# Yeni: ölçüm sonuçlarını saklayan global değişkenler
current_nW = 0
current_nH = 0

def find_marlin_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        vid = f"{port.vid:04X}" if port.vid else ""
        pid = f"{port.pid:04X}" if port.pid else ""
        if (vid == CH340_VID and pid == CH340_PID) or (vid == MEGA_VID and pid == MEGA_PID):
            return port.device
    return None

def open_serial_connection():
    global ser
    if ser is None or not ser.is_open:
        port = find_marlin_port()
        if port:
            try:
                ser = serial.Serial(port, 250000, timeout=2)
                time.sleep(2)
                print(f"Bağlantı sağlandı: {port}")
                return True
            except Exception as e:
                messagebox.showerror("Bağlantı Hatası", str(e))
                return False
        else:
            messagebox.showerror("Hata", "Marlin cihazı bulunamadı!")
            return False
    return True

def send_g28():
    if not open_serial_connection():
        return

    log_text.delete('1.0', tk.END)
    try:
        ser.reset_input_buffer()
        ser.write("G28 X Y Z\n".encode('utf-8'))
        log_text.insert(tk.END, "Komut gönderildi: G28 X Y Z\n")
        time.sleep(0.2)
        while ser.in_waiting:
            response = ser.readline().decode('utf-8', errors='ignore').strip()
            log_text.insert(tk.END, f"{response}\n")
    except Exception as e:
        messagebox.showerror("Hata", str(e))

def send_custom_command():
    command = custom_command_entry.get()
    if not command:
        messagebox.showwarning("Uyarı", "Bir komut girin!")
        return

    if not open_serial_connection():
        return

    log_text.delete('1.0', tk.END)
    try:
        ser.reset_input_buffer()
        ser.write(f"{command}\n".encode('utf-8'))
        log_text.insert(tk.END, f"Komut gönderildi: {command}\n")
        time.sleep(0.2)
        while ser.in_waiting:
            response = ser.readline().decode('utf-8', errors='ignore').strip()
            log_text.insert(tk.END, f"{response}\n")
    except Exception as e:
        messagebox.showerror("Hata", str(e))

def capture_video():
    global cap, frame
    while cap.isOpened():
        ret, new_frame = cap.read()
        if ret:
            frame = new_frame
        time.sleep(0.05)

def update_video_frame():
    global frame, current_nW, current_nH
    if frame is not None:
        scale = 26 / 10
        wP = int(210 * scale)
        hP = int(286 * scale)
        imgContours, conts = cm.getContours(frame, minArea=50000, filter=4)
        if len(conts) != 0:
            biggest = conts[0][2]
            imgWarp = cm.warpImg(frame, biggest, wP, hP)

            imgContours2, conts2 = cm.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)
            if len(conts2) != 0:
                for obj in conts2:
                    cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                    nPoints = cm.reorder(obj[2])
                    nW = round((cm.findDis(nPoints[0][0], nPoints[1][0]) / scale) / 10, 1)
                    nH = round((cm.findDis(nPoints[0][0], nPoints[2][0]) / scale) / 10, 1)

                    x, y, w, h = obj[3]
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                    (nPoints[1][0][0], nPoints[1][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                    (nPoints[2][0][0], nPoints[2][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)
                    cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)
                    cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)

                    dimensions_label.config(text=f"En: {nW}cm, Boy: {nH}cm")

                    # Yeni: ölçümleri sakla (otomatik gönderim yok)
                    current_nW = nW
                    current_nH = nH

            img_show = cv2.resize(imgContours2, (320, 240), interpolation=cv2.INTER_AREA)
        else:
            img_show = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

        img_pil = Image.fromarray(img_show)
        img_tk = ImageTk.PhotoImage(img_pil)
        video_label.img_tk = img_tk
        video_label.config(image=img_tk)

    video_label.after(10, update_video_frame)

def send_m42_ramp(start=0, end=45.174, step=5, delay=0.2):
    if not open_serial_connection():
        return
    try:
        current = start
        while current <= end:
            cmd = f"M42 P6 S{current:.3f}"
            ser.write((cmd + "\n").encode('utf-8'))
            log_text.insert(tk.END, f"Komut gönderildi: {cmd}\n")
            ser.flush()
            time.sleep(delay)
            current += step
    except Exception as e:
        messagebox.showerror("Hata", str(e))

def send_m42_ramp_down(start=45.174, end=0, step=5, delay=0.2):
    if not open_serial_connection():
        return
    try:
        current = start
        while current >= end:
            cmd = f"M42 P6 S{current:.3f}"
            ser.write((cmd + "\n").encode('utf-8'))
            log_text.insert(tk.END, f"Komut gönderildi: {cmd}\n")
            ser.flush()
            time.sleep(delay)
            current -= step
    except Exception as e:
        messagebox.showerror("Hata", str(e))
def send_autonomous():
    global current_nW, current_nH
    if not open_serial_connection():
        return
    try:
        ser.reset_input_buffer()
        ser.write("M211 S0\n".encode('utf-8'))      #soft Endstop devre dışı bırakılır
        log_text.insert(tk.END, "Komut gönderildi: M211 S0\n")

        gcode_command = f"G1 X{102 - (current_nW - 3.8) * 4.8} Y{74.2 - (current_nH - 4) * 4.945}"
        ser.write(f"{gcode_command}\n".encode('utf-8'))
        log_text.insert(tk.END, f"Komut gönderildi: {gcode_command}\n")
        ser.write("M400\n".encode('utf-8'))  # Bekle


        ser.write("G1 Z-320\n".encode('utf-8'))
        log_text.insert(tk.END, "Komut gönderildi: G1 Z-330\n")
        ser.write("M400\n".encode('utf-8'))  # Bekle

        # M42 P6 Sa  a degerinin kademeli olarak değişmesini sağlar.
        send_m42_ramp(start=0, end=34.28, step=5, delay=0.2)

        ser.write("G1 Z-100\n".encode('utf-8'))
        log_text.insert(tk.END, "Komut gönderildi: G1 Z-100\n")
        ser.write("M400\n".encode('utf-8'))  # Bekle

        ser.write("G1 Z-320\n".encode('utf-8'))
        log_text.insert(tk.END, "Komut gönderildi: G1 Z-330\n")
        ser.write("M400\n".encode('utf-8'))  # Bekle

        ser.write("M42 P6 S0\n".encode('utf-8'))
        log_text.insert(tk.END, "Komut gönderildi: M42 P6 S0\n")

        # Başlangıç konumuna dön
        ser.write("G1 X0 Y0 Z0\n".encode('utf-8'))
        log_text.insert(tk.END, "Komut gönderildi: G1 X0 Y0 Z0\n")
        ser.write("M400\n".encode('utf-8'))  # Bekle

    except Exception as e:
        messagebox.showerror("Hata", str(e))

def send_measurement_as_gcode():
    global current_nW, current_nH
    gcode_command = f"G1 X{102 - (current_nW - 3.8) *4.8} Y{74.2 - (current_nH - 4)*4.945}"
    if open_serial_connection():
        try:
            ser.reset_input_buffer()
            ser.write(f"{gcode_command}\n".encode('utf-8'))
            log_text.insert(tk.END, f"Komut gönderildi: {gcode_command}\n")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

def on_camera_select(event):
    global cap
    selected_camera_index = camera_combobox.current()
    cap = cv2.VideoCapture(selected_camera_index)
    update_video_frame()

def reset_command_flag():
    global already_sent
    already_sent = False
    log_text.insert(tk.END, "Yeni ölçüm için komut gönderimi sıfırlandı.\n")
def send_pressure_command():
    selected_label = pressure_combobox.get()
    if not selected_label:
        messagebox.showwarning("Uyarı", "Bir basınç değeri seçin!")
        return

    s_value = pressure_values.get(selected_label)
    if s_value is None:
        messagebox.showerror("Hata", "Geçersiz seçim!")
        return

    gcode_command = f"M42 P6 S{int(s_value)}"
    if open_serial_connection():
        try:
            ser.reset_input_buffer()
            ser.write(f"{gcode_command}\n".encode('utf-8'))
            log_text.insert(tk.END, f"Komut gönderildi: {gcode_command}\n")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

# Basınç değerleri: gösterilecek metin -> gönderilecek sayısal değer
pressure_values = {
    "1.4 bar": 80,
    "1.2 bar": 68.571,
    "1.0 bar": 51.143,
    "0.8 bar": 45.714,
    "0.6 bar": 34.28,
    "0.4 bar": 22.86,
    "0 bar": 0
}
# Arayüz
root = tk.Tk()
root.geometry("800x600")
def list_cameras():
    available_cameras = []
    for i in range(5):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            available_cameras.append(i)
            cap_test.release()
    return available_cameras

cameras = list_cameras()

camera_label = tk.Label(root, text="Kamera Seçin:")
camera_label.place(x=720, y=25)
camera_combobox = ttk.Combobox(root, values=cameras)
camera_combobox.place(x=690, y=5)
camera_combobox.bind("<<ComboboxSelected>>", on_camera_select)

video_label = tk.Label(root)
video_label.pack(pady=10)

dimensions_label = tk.Label(root, text="En: 0px, Boy: 0px")
dimensions_label.pack(pady=10)

tk.Label(root, text="Cihaz algılandıktan sonra home için butona basılmalı.").pack(pady=10)
tk.Button(root, text="G28 X Y Z Gönder", command=send_g28, width=25).pack(pady=10)

tk.Label(root, text="G-code Komutu Girin:").pack(pady=10)
custom_command_entry = tk.Entry(root, width=40)
custom_command_entry.pack(pady=5)
tk.Button(root, text="Komut Gönder", command=send_custom_command, width=25).pack(pady=10)


# Yeni: ölçülen değeri G-code olarak gönder butonu
tk.Button(root, text="Ölçülen Değeri G-code Olarak Gönder", command=send_measurement_as_gcode, width=30).pack(pady=10)
tk.Button(root, text="Otonom çalışmayı başlat", command=send_autonomous, width=30).place(x=970, y=290)

log_text = scrolledtext.ScrolledText(root, width=600, height=15)
log_text.pack(padx=10, pady=10)

tk.Label(root, text="Basınç Değeri Seçin (M42 P6 Sxxx):").place(x=975, y=185)
pressure_combobox = ttk.Combobox(root, values=list(pressure_values.keys()), state="readonly", width=20)
pressure_combobox.place(x=995, y=210)
# Yeni: Gönder butonu
tk.Button(root, text="Basınç Komutunu Gönder", command=send_pressure_command, width=30).place(x=970, y=240)
cap = cv2.VideoCapture(cameras[1] if len(cameras) > 1 else 0)

def start_video_thread():
    capture_video()

video_thread = threading.Thread(target=start_video_thread)
video_thread.daemon = True
video_thread.start()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
