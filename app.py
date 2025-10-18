import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import joblib
from collections import deque
import os
import warnings 

# ==============================
# CẤU HÌNH CƠ BẢN
# ==============================
MODEL_PATH = "softmax_model_best1.pkl" # Chứa W, b, classes
SCALER_PATH = "scale1.pkl"              # Chứa X_mean, X_std
LABEL_MAP_PATH = "label_map_5cls.json"

SMOOTH_WINDOW = 3 # Giữ 3 để tăng độ nhạy
EPS = 1e-8 
NEW_WIDTH, NEW_HEIGHT = 640, 480 # Kích thước khung hình sau khi resize

# NGƯỠNG EAR CỨNG: ĐÃ LOẠI BỎ THEO YÊU CẦU

# ==============================
# HÀM DỰ ĐOÁN SOFTMAX
# ==============================
def softmax_predict(X, W, b):
    """Thực hiện dự đoán Softmax."""
    # X phải là (N, 9)
    logits = X @ W + b
    return np.argmax(logits, axis=1)

@st.cache_resource
def load_assets():
    """Tải tham số W, b, scaler (mean, std) và label map"""
    try:
        # 1. Tải mô hình Softmax (W và b)
        with open(MODEL_PATH, "rb") as f:
            model_data = joblib.load(f)
            W = model_data["W"]
            b = model_data["b"]

        # 2. Tải scaler (mean và std)
        with open(SCALER_PATH, "rb") as f:
            scaler_data = joblib.load(f)
            mean_data = scaler_data["X_mean"]
            std_data = scaler_data["X_std"]
            
        # Kiểm tra kích thước đặc trưng
        EXPECTED_FEATURE_SIZE = 9
        if W.shape[0] != EXPECTED_FEATURE_SIZE:
             st.error(f"LỖI KHÔNG TƯƠNG THÍCH: Mô hình yêu cầu {W.shape[0]} đặc trưng, nhưng ứng dụng này chỉ trích xuất {EXPECTED_FEATURE_SIZE} đặc trưng tức thời.")
             st.stop()


        # 3. Tải label map
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        id2label = {int(v): k for k, v in label_map.items()}

        return W, b, mean_data, std_data, id2label

    except FileNotFoundError as e:
        st.error(f"LỖỖI FILE: Không tìm thấy file tài nguyên. Vui lòng kiểm tra đường dẫn: {e.filename}")
        st.stop()
    except KeyError as e:
        st.error(f"LỖỖI CẤU TRÚC FILE: Kiểm tra cấu trúc file model/scaler (thiếu key: {e}).")
        st.stop()
    except Exception as e:
        st.error(f"LỖỖI LOAD DỮ LIỆU: File tài nguyên bị hỏng (corrupted) hoặc không thể giải mã. Chi tiết: {e}")
        st.stop()

# Tải tài sản (Chạy một lần)
W, b, mean, std, id2label = load_assets()
classes = list(id2label.values())

# ----------------------------------------------------------------------
## HÀM TÍNH ĐẶC TRƯNG
# ----------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
EYE_LEFT_IDX = np.array([33, 159, 145, 133, 153, 144])
EYE_RIGHT_IDX = np.array([362, 386, 374, 263, 380, 385])
MOUTH_IDX = np.array([61, 291, 0, 17, 78, 308])

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

def get_extra_features(landmarks):
    nose, chin = landmarks[1], landmarks[152]
    angle_pitch_extra = np.degrees(np.arctan2(chin[1] - nose[1], (chin[2] - nose[2]) + EPS))
    forehead_y = np.mean(landmarks[[10, 338, 297, 332, 284], 1])
    cheek_dist = np.linalg.norm(landmarks[50] - landmarks[280])
    return angle_pitch_extra, forehead_y, cheek_dist


# ----------------------------------------------------------------------
## WEBRTC VIDEO PROCESSOR (Logic xử lý Real-time)
# ----------------------------------------------------------------------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.W = W
        self.b = b
        self.mean = mean
        self.std = std
        self.id2label = id2label
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.pred_queue = deque(maxlen=SMOOTH_WINDOW)
        self.last_pred_label = "CHO DU LIEU VAO"
        self.N_FEATURES = 9 

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_array = frame.to_ndarray(format="bgr24")

        # 1. RESIZE KHUNG HÌNH (Tăng tốc độ)
        frame_resized = cv2.resize(frame_array, (NEW_WIDTH, NEW_HEIGHT))
        h, w = frame_resized.shape[:2]

        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # Flip để gương mặt khớp với tọa độ (như trong code test cam desktop)
        rgb_flipped = cv2.flip(rgb, 1) 
        
        results = self.face_mesh.process(rgb_flipped)
        
        current_pred_label = "unknown"

        # --- 2. TRÍCH XUẤT 9 ĐẶC TRƯNG TỨC THỜI ---
        if results.multi_face_landmarks:
            landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark])

            # Tính toán 9 đặc trưng tức thời
            ear_l = eye_aspect_ratio(landmarks, True)
            ear_r = eye_aspect_ratio(landmarks, False)
            mar = mouth_aspect_ratio(landmarks)
            yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
            angle_pitch_extra, forehead_y, cheek_dist = get_extra_features(landmarks)
            
            # ear_avg = (ear_l + ear_r) / 2.0 -> KHÔNG CẦN NỮA

            # Mảng 9 đặc trưng
            feats = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                              angle_pitch_extra, forehead_y, cheek_dist], dtype=np.float32)

            # --- 3. DỰ ĐOÁN SOFTMAX ---
            
            # Chuẩn hóa chỉ trên 9 đặc trưng
            feats_scaled = (feats - self.mean[:self.N_FEATURES]) / (self.std[:self.N_FEATURES] + EPS)
            
            # Dự đoán Softmax
            pred_idx = softmax_predict(np.expand_dims(feats_scaled, axis=0), self.W, self.b)[0]
            pred_label = self.id2label.get(pred_idx, f"Class {pred_idx}")
            
            # Thêm vào queue làm mượt
            self.pred_queue.append(pred_label)

        # --- 4. SMOOTHING VÀ HIỂN THỊ KẾT QUẢ ---
        if len(self.pred_queue) > 0:
            # Lấy nhãn xuất hiện nhiều nhất trong cửa sổ làm mượt
            self.last_pred_label = max(set(self.pred_queue), key=self.pred_queue.count)
        
        
        cv2.putText(frame_resized, f"Trang thai: {self.last_pred_label.upper()}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)
        # LOẠI BỎ HIỂN THỊ EAR AVG
        # cv2.putText(frame_resized, f"EAR avg: {ear_avg:.2f}", (10, 110), ...)

        # BỎ THAO TÁC LẬT LẦN 2
        frame_display = frame_resized 
        return av.VideoFrame.from_ndarray(frame_display, format="bgr24")

# ----------------------------------------------------------------------
## GIAO DIỆN STREAMLIT CHÍNH
# ----------------------------------------------------------------------
st.set_page_config(page_title="Demo Softmax - 9 Đặc trưng Tức thời", layout="wide")
st.title("🧠 Nhận diện trạng thái mất tập trung bằng 9 đặc trưng khuôn mặt (Thời gian thực)")
st.success(f"Mô hình sẵn sàng! Các nhãn: {classes} | Cần 9 đặc trưng.")
st.warning("Vui lòng chấp nhận yêu cầu truy cập camera từ trình duyệt của bạn.")
st.markdown("---")

# === Đã thêm cấu trúc cột để căn giữa và thu hẹp màn hình video ===
col1, col2, col3 = st.columns([1, 4, 1]) # Tỷ lệ 1:4:1 giúp căn giữa video

with col2: # Đặt component vào cột giữa
    webrtc_streamer(
        key="softmax_driver_live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=DrowsinessProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
