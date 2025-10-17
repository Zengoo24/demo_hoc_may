import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque

# Cấu hình trang Streamlit
st.set_page_config(layout="wide")

# ==============================
# CẤU HÌNH & HÀM TIỆN ÍCH
# ==============================
# Các hằng số được sử dụng trong script gốc
MODEL_PATH = "softmax_model_best.pkl"
SCALER_PATH = "scale.pkl"
SMOOTH_WINDOW = 8
CONF_THRESHOLD = 0.5
FPS_SMOOTH = 0.9
EXPECTED_FEATURE_SIZE = 9
EPS = 1e-6

# ==============================
# CACHING CÁC TÀI NGUYÊN NẶNG (Mô hình và Mediapipe)
# ==============================
@st.cache_resource
def load_model_and_scaler():
    """Tải mô hình Softmax và dữ liệu chuẩn hóa chỉ một lần."""
    try:
        # Load Model
        model_data = joblib.load(MODEL_PATH)
        W = model_data["W"]
        b = model_data["b"]
        CLASSES = model_data["classes"]
        idx2label = {i: lbl for i, lbl in enumerate(CLASSES)}

        # Load Scaler
        scaler_data = joblib.load(SCALER_PATH)
        X_mean = scaler_data["X_mean"]
        X_std = scaler_data["X_std"]
        
        # Kiểm tra kích thước đặc trưng
        if len(X_mean) != EXPECTED_FEATURE_SIZE:
            st.error(f"Lỗi: Kích thước đặc trưng của scaler ({len(X_mean)}) không khớp với kỳ vọng ({EXPECTED_FEATURE_SIZE}).")
            return None, None, None, None, None, None
            
        st.success(f"✅ Đã load Model Softmax. Các trạng thái: {CLASSES}")
        return W, b, CLASSES, X_mean, X_std, idx2label
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file mô hình '{MODEL_PATH}' hoặc '{SCALER_PATH}'. Hãy đảm bảo các file này nằm trong cùng thư mục với app.py.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Lỗi khi tải tài nguyên: {e}")
        return None, None, None, None, None, None

@st.cache_resource
def get_face_mesh_object():
    """Tải và cache đối tượng MediaPipe FaceMesh chỉ một lần."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    # Drawing Spec được dùng nếu muốn vẽ landmarks
    drawing_spec = mp_face_mesh.DrawingSpec(thickness=1, circle_radius=1)
    return face_mesh, drawing_spec

# ==============================
# KHAI BÁO CÁC CHỈ SỐ LANDMARK
# ==============================
EYE_LEFT_IDX = np.array([33, 159, 145, 133, 153, 144])
EYE_RIGHT_IDX = np.array([362, 386, 374, 263, 380, 385])
MOUTH_IDX = np.array([61, 291, 0, 17, 78, 308])

# ==============================
# HÀM DỰ ĐOÁN & TÍNH ĐẶC TRƯNG
# ==============================
def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def predict_proba(x, W, b):
    # Đảm bảo x là mảng 2D (1, feature_size)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    z = np.dot(x, W) + b
    return softmax(z)

def eye_aspect_ratio(landmarks, left=True):
    idx = EYE_LEFT_IDX if left else EYE_RIGHT_IDX
    pts = landmarks[idx, :2]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def mouth_aspect_ratio(landmarks):
    pts = landmarks[MOUTH_IDX, :2]
    A = np.linalg.norm(pts[0] - pts[1])
    B = np.linalg.norm(pts[4] - pts[5])
    C = np.linalg.norm(pts[2] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def head_pose_yaw_pitch_roll(landmarks):
    left_eye = landmarks[33][:2]
    right_eye = landmarks[263][:2]
    nose = landmarks[1][:2]
    chin = landmarks[152][:2]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll = np.degrees(np.arctan2(dy, dx + EPS))
    interocular = np.linalg.norm(right_eye - left_eye) + EPS
    eyes_center = (left_eye + right_eye) / 2.0
    yaw = np.degrees(np.arctan2((nose[0] - eyes_center[0]), interocular))
    baseline = chin - eyes_center
    pitch = np.degrees(np.arctan2((nose[1] - eyes_center[1]), (np.linalg.norm(baseline) + EPS)))
    return yaw, pitch, roll


# ==============================
# HÀM CHÍNH CỦA STREAMLIT
# ==============================
def main():
    st.title("👨‍💻 Demo Giám sát Tài xế (DMS) - Streamlit Softmax")
    st.markdown("""
    Ứng dụng này sử dụng **MediaPipe Face Mesh** và mô hình **Softmax Regression** để dự đoán trạng thái của tài xế.
    
    ⚠️ **Lưu ý quan trọng:** Ứng dụng này sử dụng `cv2.VideoCapture(0)` và **chỉ hoạt động khi chạy cục bộ** (trên máy tính của bạn) và cần có webcam. Nó sẽ KHÔNG chạy trên các dịch vụ cloud như Streamlit Community Cloud do hạn chế truy cập camera.
    """)
    
    # Tải mô hình và Mediapipe (Chỉ chạy 1 lần)
    W, b, CLASSES, X_mean, X_std, idx2label = load_model_and_scaler()
    face_mesh, _ = get_face_mesh_object()

    if W is None:
        return # Dừng nếu load model thất bại

    # 1. KHỞI TẠO STATE
    if 'run' not in st.session_state:
        st.session_state.run = False
    if 'pred_queue' not in st.session_state:
        st.session_state.pred_queue = deque(maxlen=SMOOTH_WINDOW)
    if 'pTime' not in st.session_state:
        st.session_state.pTime = time.time()
    if 'fps' not in st.session_state:
        st.session_state.fps = 0

    # 2. KHU VỰC ĐIỀU KHIỂN
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.session_state.run:
            stop_button = st.button("🔴 Dừng Camera", key="stop", type="primary")
            if stop_button:
                st.session_state.run = False
                st.session_state.pred_queue = deque(maxlen=SMOOTH_WINDOW)
                # Dùng time.sleep ngắn để đảm bảo vòng lặp kết thúc
                time.sleep(0.1) 
                st.rerun()
        else:
            start_button = st.button("🟢 Bắt đầu Camera", key="start", type="primary")
            if start_button:
                st.session_state.run = True
                st.session_state.pTime = time.time()
                st.session_state.pred_queue = deque(maxlen=SMOOTH_WINDOW)
                st.rerun()

    # 3. VÒNG LẶP VIDEO VÀ HIỂN THỊ
    
    # st.empty() tạo một placeholder để cập nhật liên tục frame video
    st_frame = st.empty()
    st_status = st.empty()
    
    if st.session_state.run:
        # Mở camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st_frame.error("❌ Không thể truy cập Webcam. Hãy đảm bảo webcam không được sử dụng bởi ứng dụng khác và đã cấp quyền.")
            st.session_state.run = False
            return
            
        st_status.info("▶️ Đang chạy. Nhấn 'Dừng Camera' để thoát.")

        # Lấy lại các biến state
        pred_queue = st.session_state.pred_queue
        pTime = st.session_state.pTime
        fps = st.session_state.fps
        
        # Vòng lặp xử lý frame
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st_frame.error("Lỗi đọc khung hình từ Camera.")
                st.session_state.run = False
                break

            # Xử lý frame
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            current_pred_label = "unknown"
            current_pred_conf = 0.0
            
            color_bgr = (0, 0, 255) # Mặc định BGR: Đỏ

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                # Lấy tọa độ landmark, nhân với kích thước frame
                landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in face.landmark])

                # 1. TRÍCH XUẤT 9 ĐẶC TRƯNG TỨC THỜI
                ear_l = eye_aspect_ratio(landmarks, True)
                ear_r = eye_aspect_ratio(landmarks, False)
                mar = mouth_aspect_ratio(landmarks)
                yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
                nose, chin = landmarks[1], landmarks[152]
                angle_pitch_extra = np.degrees(np.arctan2(chin[1] - nose[1], (chin[2] - nose[2]) + EPS))
                forehead_y = np.mean(landmarks[[10, 338, 297, 332, 284], 1])
                cheek_dist = np.linalg.norm(landmarks[50] - landmarks[280])

                feats = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                                  angle_pitch_extra, forehead_y, cheek_dist], dtype=np.float32)

                # 2. CHUẨN HÓA (Scaling)
                feats_scaled = (feats - X_mean[:9]) / (X_std[:9] + EPS)

                # 3. DỰ ĐOÁN Softmax
                probs = predict_proba(feats_scaled, W, b)
                probs = np.array(probs).flatten()
                pred_idx = np.argmax(probs)
                current_pred_conf = float(probs[pred_idx])
                current_pred_label = idx2label[pred_idx]

                # 4. LÀM MƯỢT KẾT QUẢ DỰ ĐOÁN
                if current_pred_conf > CONF_THRESHOLD:
                    pred_queue.append(current_pred_label)
                else:
                    pred_queue.append("unknown")


            # ======== SMOOTH PREDICTION ========
            if len(pred_queue) > 0 and pred_queue.count("unknown") < len(pred_queue):
                # Lọc bỏ "unknown" trước khi tìm nhãn phổ biến nhất
                valid_preds = [p for p in pred_queue if p != "unknown"]
                if valid_preds:
                    final_label = max(set(valid_preds), key=valid_preds.count)
                else:
                    final_label = "unknown"
            else:
                final_label = "unknown"

            # Xác định màu hiển thị (BGR)
            label_lower = final_label.lower()
            if "drowsy" in label_lower or "sleep" in label_lower:
                color_bgr = (0, 0, 255)       # Đỏ
            elif "distract" in label_lower:
                color_bgr = (0, 255, 255)     # Vàng
            elif "awake" in label_lower or "calm" in label_lower:
                color_bgr = (255, 0, 0)       # Xanh dương
            else:
                color_bgr = (255, 255, 255)   # Trắng


            # ======== HIỂN THỊ FPS ========
            cTime = time.time()
            fps = FPS_SMOOTH * fps + (1 - FPS_SMOOTH) * (1 / (cTime - pTime + EPS))
            pTime = cTime

            # Vẽ chữ lên frame (OpenCV BGR)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"State: {final_label} ({current_pred_conf:.2f})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 3)

            # Cập nhật state trước khi Streamlit reruns
            st.session_state.pTime = pTime
            st.session_state.fps = fps
            st.session_state.pred_queue = pred_queue

            # HIỂN THỊ FRAME TRONG STREAMLIT
            # Chuyển từ BGR (OpenCV) sang RGB (Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Kết thúc vòng lặp
        cap.release()
        st_status.warning("🛑 Camera đã dừng.")

# Chạy hàm chính
if __name__ == '__main__':
    main()
