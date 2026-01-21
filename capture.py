import cv2
import os
import time
import numpy as np

# --- CẤU HÌNH ---
SAVE_DIR = "resistor_dataset"  # Tên thư mục lưu ảnh
CAP_WIDTH = 1280               # Cố gắng đặt độ phân giải cao nhất của cam
CAP_HEIGHT = 720

# Tạo thư mục lưu nếu chưa có
os.makedirs(SAVE_DIR, exist_ok=True)

def enhance_image(image):
    """
    Hàm xử lý ảnh chuyên sâu cho linh kiện điện tử:
    1. Khử nhiễu (Denoise) để ảnh mịn hơn.
    2. Làm nét (Sharpen) để các vạch màu điện trở rõ ràng hơn.
    """
    # 1. Khử nhiễu: fastNlMeansDenoisingColored rất tốt để loại bỏ nhiễu hạt
    # h=10: Sức mạnh lọc (cao hơn thì mịn hơn nhưng mất chi tiết, 10 là vừa vặn)
    clean_img = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # 2. Làm nét: Sử dụng kernel convolution để tăng độ tương phản cạnh
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened_img = cv2.filter2D(clean_img, -1, kernel_sharpen)

    return sharpened_img

def main():
    # Khởi tạo camera
    cap = cv2.VideoCapture(0) # Số 0 thường là webcam mặc định
    
    # Thiết lập độ phân giải (Càng cao chụp resistor càng rõ)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    # Kiểm tra cam
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return

    print("--- HƯỚNG DẪN ---")
    print("👉 Nhấn phím 'SPACE' (Cách) để chụp ảnh.")
    print("👉 Nhấn phím 'q' để thoát.")
    print(f"Ảnh sẽ được lưu vào thư mục: {SAVE_DIR}")

    img_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi đọc từ camera.")
            break

        # Hiển thị khung hình trực tiếp (Live view)
        # Lưu ý: Ta hiển thị ảnh gốc để không bị lag, chỉ xử lý khi lưu
        cv2.imshow('Camera - Resistor Capture', frame)

        key = cv2.waitKey(1) & 0xFF

        # Nút Chụp (SPACE)
        if key == ord('z'):
            print("Đang xử lý ảnh... vui lòng đợi...")
            
            # --- BẮT ĐẦU QUY TRÌNH XỬ LÝ ẢNH ---
            processed_frame = enhance_image(frame)
            
            # Tạo tên file theo thời gian thực để không trùng
            timestamp = int(time.time())
            filename = os.path.join(SAVE_DIR, f"resistor_{timestamp}.jpg")
            
            # Lưu ảnh đã xử lý
            cv2.imwrite(filename, processed_frame)
            
            print(f"✅ Đã lưu: {filename} (Đã khử nhiễu & làm nét)")
            img_count += 1
            
            # Tạm dừng 0.5 giây để tránh chụp đúp (Tốc độ vừa phải)
            time.sleep(0.5) 

        # Nút Thoát (q)
        elif key == ord('q'):
            print("Đang thoát chương trình...")
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()