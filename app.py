import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import joblib
from collections import deque
import os # Giữ lại os, mặc dù không dùng trong code hiện tại

# ==============================
# CẤU HÌNH CƠ BẢN
# ==============================
MODEL_PATH = "softmax_model_best.pkl" # Chứa W, b, classes
SCALER_PATH = "scale.pkl"              # Chứa X_mean, X_std
LABEL_MAP_PATH = "label_map.json"

SMOOTH_WINDOW = 8 # Cửa sổ làm mượt (smoothing)
EPS = 1e-8        # Số epsilon nhỏ để tránh chia cho 0
WINDOW_SIZE = 15  # Cửa sổ khung hình để tính đặc trưng thống kê
NEW_WIDTH, NEW_HEIGHT = 640, 480 # Kích thước khung hình sau khi resize

# ==============================
# HÀM DỰ ĐOÁN SOFTMAX
# ==============================
def softmax_predict(X, W, b):
    """Thực hiện dự đoán Softmax."""
    # X: Đặc trưng đã được scale (shape: (1, n_features))
    # W: Ma trận trọng số (shape: (n_features, n_classes))
    # b: Vector bias (shape: (1, n_classes))
    logits = X @ W + b
    # Trả về chỉ mục lớp có xác suất cao nhất
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

        # 3. Tải label map
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        # Chuyển đổi key từ string sang int (vì JSON lưu key là string)
        id2label = {int(v): k for k, v in label_map.items()}

        return W, b, mean_data, std_data, id2label

    except FileNotFoundError as e:
        st.error(f"Lỗi File: Không tìm thấy file tài nguyên. Vui lòng kiểm tra đường dẫn: {e.filename}")
        st.stop()
    except KeyError as e:
        st.error(f"Lỗi Key: Kiểm tra cấu trúc file model/scaler (thiếu key: {e}).")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi Load: {e}")
        st.stop()

# Tải tài sản (Chạy một lần khi khởi động ứng dụng)
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
    # chin = landmarks[152][:2] # Không cần chin để tính yaw/roll cơ bản
    
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    # Roll (Độ nghiêng của đầu)
    roll = np.degrees(np.arctan2(dy, dx + EPS))
    
    interocular = np.linalg.norm(right_eye - left_eye) + EPS
    eyes_center = (left_eye + right_eye) / 2.0
    
    # Yaw (Xoay ngang - quay trái/phải)
    # Lấy sự khác biệt x giữa mũi và tâm mắt, chuẩn hóa bằng khoảng cách giữa hai mắt.
    yaw = np.degrees(np.arctan2((nose[0] - eyes_center[0]), interocular))
    
    # Pitch (Gật đầu lên/xuống)
    # Sử dụng sự khác biệt y giữa mũi và tâm mắt
    pitch = np.degrees(np.arctan2((nose[1] - eyes_center[1]), interocular))
    
    # Góc pitch trong code cũ của bạn có sử dụng `chin`, tôi thay bằng công thức đơn giản hơn
    # Nếu code huấn luyện gốc của bạn dùng chin, hãy giữ lại công thức đó.
    # baseline = chin - eyes_center
    # pitch = np.degrees(np.arctan2((nose[1] - eyes_center[1]), (np.linalg.norm(baseline) + EPS)))
    
    return yaw, pitch, roll

def get_extra_features(landmarks):
    nose, chin = landmarks[1], landmarks[152]
    # Lưu ý: Tính toán z-coord có thể không ổn định
    angle_pitch_extra = np.degrees(np.arctan2(chin[1] - nose[1], (chin[2] - nose[2]) + EPS))
    forehead_y = np.mean(landmarks[[10, 338, 297, 332, 284], 1])
    cheek_dist = np.linalg.norm(landmarks[50] - landmarks[280])
    return angle_pitch_extra, forehead_y, cheek_dist


# ----------------------------------------------------------------------
## WEBRTC VIDEO PROCESSOR (Logic xử lý Real-time)
# ----------------------------------------------------------------------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        # Tải tham số Softmax từ st.cache_resource
        self.W = W
        self.b = b
        self.mean = mean
        self.std = std
        self.id2label = id2label
        
        # Khởi tạo Face Mesh (chỉ một lần)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False, # Đã đặt False như trong code của bạn
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        self.frame_queue = deque(maxlen=WINDOW_SIZE)
        self.pred_queue = deque(maxlen=SMOOTH_WINDOW)
        self.last_pred_label = "CHO DU LIEU VAO"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_array = frame.to_ndarray(format="bgr24")

        # 💡 1. RESIZE KHUNG HÌNH (Tăng tốc độ xử lý)
        frame_resized = cv2.resize(frame_array, (NEW_WIDTH, NEW_HEIGHT))
        h, w = frame_resized.shape[:2]

        # Chuyển sang RGB (dùng khung hình đã resize)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Xử lý Face Mesh
        results = self.face_mesh.process(rgb)

        # --- 2. TRÍCH XUẤT ĐẶC TRƯNG ---
        if results.multi_face_landmarks:
            # landmarks được scale theo kích thước mới (640, 480)
            landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark])

            # Tính 9 đặc trưng cơ bản
            ear_l = eye_aspect_ratio(landmarks, True)
            ear_r = eye_aspect_ratio(landmarks, False)
            mar = mouth_aspect_ratio(landmarks)
            yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
            angle_pitch_extra, forehead_y, cheek_dist = get_extra_features(landmarks)

            feat = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                              angle_pitch_extra, forehead_y, cheek_dist], dtype=np.float32)
            self.frame_queue.append(feat)

            # --- 3. DỰ ĐOÁN KHI ĐỦ KHUNG HÌNH ---
            if len(self.frame_queue) == WINDOW_SIZE:
                window = np.array(self.frame_queue)

                # Tính 24 đặc trưng (Không thay đổi logic của bạn)
                mean_feats = window.mean(axis=0)
                std_feats = window.std(axis=0)
                yaw_diff = np.mean(np.abs(np.diff(window[:, 3])))
                pitch_diff = np.mean(np.abs(np.diff(window[:, 4])))
                roll_diff = np.mean(np.abs(np.diff(window[:, 5])))
                mar_mean = np.mean(window[:, 2])
                ear_mean = np.mean((window[:, 0] + window[:, 1]) / 2.0)
                mar_ear_ratio = mar_mean / (ear_mean + EPS)
                yaw_pitch_ratio = np.mean(np.abs(window[:, 3])) / (np.mean(np.abs(window[:, 4])) + EPS)
                
                # Nối 24 đặc trưng
                feats_24 = np.concatenate([mean_feats, std_feats, [yaw_diff, pitch_diff, roll_diff, np.max(window[:, 2]), mar_ear_ratio, yaw_pitch_ratio]])

                # Chuẩn hóa, Dự đoán
                feats_scaled = (feats_24 - self.mean) / (self.std + EPS) # Thêm EPS để tránh chia cho 0
                pred_idx = softmax_predict(np.expand_dims(feats_scaled, axis=0), self.W, self.b)[0]
                
                pred_label = self.id2label.get(pred_idx, f"Class {pred_idx}")
                self.pred_queue.append(pred_label)

                # Xóa khung hình cũ (overlap)
                FRAMES_TO_DELETE = 5
                for _ in range(FRAMES_TO_DELETE):
                    if self.frame_queue:
                        self.frame_queue.popleft()

        # --- 4. SMOOTHING VÀ HIỂN THỊ KẾT QUẢ ---
        if len(self.pred_queue) > 0:
            # Chọn nhãn xuất hiện nhiều nhất trong cửa sổ làm mượt
            self.last_pred_label = max(set(self.pred_queue), key=self.pred_queue.count)
        
        # Chèn văn bản vào khung hình đã resize
        cv2.putText(frame_resized, f"Trang thai: {self.last_pred_label.upper()}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)

        # Trả về khung hình đã resize
        return av.VideoFrame.from_ndarray(frame_resized, format="bgr24")

# ----------------------------------------------------------------------
## GIAO DIỆN STREAMLIT CHÍNH
# ----------------------------------------------------------------------
st.set_page_config(page_title="Demo Softmax", layout="wide")
st.title("🧠 Nhận diện trạng thái mất tập trung bằng mô hình học máy.")
st.success(f"Mô hình sẵn sàng! Các nhãn: {classes}")
st.warning("Vui lòng chấp nhận yêu cầu truy cập camera từ trình duyệt của bạn.")
st.markdown("---")


# Khởi tạo WebRTC Streamer
webrtc_streamer(
    key="softmax_driver_live",
    mode=WebRtcMode.SENDRECV,
    # Cấu hình STUN servers để thiết lập kết nối
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False}, # Chỉ bật video
    async_processing=True, # Cho phép xử lý không đồng bộ (tăng tốc độ)
)
