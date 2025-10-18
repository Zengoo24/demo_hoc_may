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
from PIL import Image

# ==============================
# CẤU HÌNH
# ==============================
MODEL_PATH = "softmax_model_best1.pkl" # CẢNH BÁO: PHẢI ĐƯỢC HUẤN LUYỆN VỚI 10 ĐẶC TRƯNG
SCALER_PATH = "scale1.pkl"              # CẢNH BÁO: PHẢI CHỨA MEAN/STD CHO 10 ĐẶC TRƯNG
LABEL_MAP_PATH = "label_map_5cls.json"

SMOOTH_WINDOW = 5 
BLINK_THRESHOLD = 0.20 # Ngưỡng cứng cho BLINK
EPS = 1e-8 
NEW_WIDTH, NEW_HEIGHT = 640, 480 
N_FEATURES = 10 # Số lượng đặc trưng mong đợi

# ==============================
# HÀM DỰ ĐOÁN SOFTMAX
# ==============================
def softmax_predict(X, W, b):
    """Thực hiện dự đoán Softmax."""
    # X phải là (N, 10)
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
            
        if W.shape[0] != N_FEATURES:
             st.error(f"LỖI KHÔNG TƯƠNG THÍCH: Mô hình yêu cầu {W.shape[0]} đặc trưng, nhưng ứng dụng này trích xuất {N_FEATURES} đặc trưng. Vui lòng kiểm tra lại file model!")
             st.stop()

        # 3. Tải label map
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        id2label = {int(v): k for k, v in label_map.items()}

        return W, b, mean_data, std_data, id2label

    except FileNotFoundError as e:
        st.error(f"LỖI FILE: Không tìm thấy file tài nguyên. Vui lòng kiểm tra đường dẫn: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"LỖI LOAD DỮ LIỆU: Chi tiết: {e}")
        st.stop()

# Tải tài sản (Chạy một lần)
W, b, mean, std, id2label = load_assets()
classes = list(id2label.values())

# ----------------------------------------------------------------------
## HÀM TÍNH ĐẶC TRƯNG (Feature Extraction Functions)
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
## HÀM XỬ LÝ ẢNH TĨNH
# ----------------------------------------------------------------------
def process_static_image(image_file, mesh, W, b, mean, std, id2label):
    # Đọc ảnh từ file uploader
    image = np.array(Image.open(image_file).convert('RGB'))
    
    # Resize ảnh để xử lý nhanh hơn và chuẩn hóa kích thước
    image_resized = cv2.resize(image, (NEW_WIDTH, NEW_HEIGHT))
    h, w = image_resized.shape[:2]
    
    # Chuẩn bị ảnh cho MediaPipe (lật để tọa độ landmarks khớp)
    image_for_mp = cv2.flip(image_resized, 1)
    
    # Xử lý MediaPipe
    results = mesh.process(image_for_mp)
    
    result_label = "Chưa tìm thấy khuôn mặt"
    
    if results.multi_face_landmarks:
        landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark])
        
        # 1. Trích xuất đặc trưng
        ear_l = eye_aspect_ratio(landmarks, True)
        ear_r = eye_aspect_ratio(landmarks, False)
        ear_avg = (ear_l + ear_r) / 2.0
        mar = mouth_aspect_ratio(landmarks)
        yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
        angle_pitch_extra, forehead_y, cheek_dist = get_extra_features(landmarks)
        
        # 2. Xử lý đặc trưng động cho ảnh tĩnh (DELTA_EAR = 0)
        delta_ear_value = 0.0 # Bằng 0 vì không có sự thay đổi theo thời gian
        
        # 3. Áp dụng luật Heuristic
        if ear_avg < BLINK_THRESHOLD:
            result_label = "BLINK (Heuristic)"
        else:
            # 4. Chạy Softmax (10 đặc trưng)
            feats = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                              angle_pitch_extra, delta_ear_value, forehead_y, cheek_dist], dtype=np.float32)

            feats_scaled = (feats - mean[:N_FEATURES]) / (std[:N_FEATURES] + EPS)
            pred_idx = softmax_predict(np.expand_dims(feats_scaled, axis=0), W, b)[0]
            result_label = id2label.get(pred_idx, "UNKNOWN")
            
        # Hiển thị kết quả:
        # Chuyển RGB sang BGR để OpenCV xử lý text
        image_display = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        
        # Lật ngược ảnh để người dùng thấy ảnh giống như ảnh gốc
        image_display_flipped = cv2.flip(image_display, 1)

        cv2.putText(image_display_flipped, f"Trang thai: {result_label.upper()}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        # Chuyển lại BGR sang RGB cho Streamlit
        final_image_rgb = cv2.cvtColor(image_display_flipped, cv2.COLOR_BGR2RGB)

        return final_image_rgb, result_label

    # Trường hợp không tìm thấy khuôn mặt
    image_display = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    cv2.putText(image_display, "KHONG TIM THAY KHUON MAT", (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image_display, result_label


# ----------------------------------------------------------------------
## WEBRTC VIDEO PROCESSOR (Logic xử lý Real-time)
# ----------------------------------------------------------------------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        # Khởi tạo các tham số và MediaPipe
        self.W = W; self.b = b; self.mean = mean; self.std = std; self.id2label = id2label
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.pred_queue = deque(maxlen=SMOOTH_WINDOW)
        self.last_pred_label = "CHO DU LIEU VAO"
        self.N_FEATURES = N_FEATURES
        self.last_ear_avg = 0.4 # Lịch sử EAR cho Delta EAR

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_array = frame.to_ndarray(format="bgr24")

        # 1. RESIZE FRAME
        frame_resized = cv2.resize(frame_array, (NEW_WIDTH, NEW_HEIGHT))
        h, w = frame_resized.shape[:2]

        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        rgb_flipped = cv2.flip(rgb, 1) # Lật ảnh chỉ cho xử lý MediaPipe
        
        results = self.face_mesh.process(rgb_flipped)
        
        delta_ear_value = 0.0
        predicted_label_frame = "UNKNOWN"

        # --- 2. TRÍCH XUẤT 10 ĐẶC TRƯNG VÀ DỰ ĐOÁN ---
        if results.multi_face_landmarks:
            landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark])

            ear_l = eye_aspect_ratio(landmarks, True); ear_r = eye_aspect_ratio(landmarks, False); ear_avg = (ear_l + ear_r) / 2.0
            
            # 1. ÁP DỤNG LUẬT HEURISTIC CỨNG CHO BLINK 
            if ear_avg < BLINK_THRESHOLD:
                predicted_label_frame = "blink"
            else:
                # 2. SỬ DỤNG SOFTMAX CHO CÁC HÀNH VI KHÁC
                
                mar = mouth_aspect_ratio(landmarks)
                yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
                angle_pitch_extra, forehead_y, cheek_dist = get_extra_features(landmarks)

                delta_ear_value = ear_avg - self.last_ear_avg 
                self.last_ear_avg = ear_avg

                feats = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                                angle_pitch_extra, delta_ear_value, forehead_y, cheek_dist], dtype=np.float32)

                feats_scaled = (feats - self.mean[:self.N_FEATURES]) / (self.std[:self.N_FEATURES] + EPS)
                pred_idx = softmax_predict(np.expand_dims(feats_scaled, axis=0), self.W, self.b)[0]
                predicted_label_frame = self.id2label.get(pred_idx, "UNKNOWN")
            
            self.pred_queue.append(predicted_label_frame)
        
        else:
             self.last_ear_avg = 0.4 

        # --- 4. SMOOTHING VÀ HIỂN THỊ KẾT QUẢ ---
        if len(self.pred_queue) > 0:
            self.last_pred_label = max(set(self.pred_queue), key=self.pred_queue.count)
        
        cv2.putText(frame_resized, f"Trang thai: {self.last_pred_label.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)
        cv2.putText(frame_resized, f"Delta EAR: {delta_ear_value:.3f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame_resized, f"EAR Threshold: <{BLINK_THRESHOLD}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(frame_resized, format="bgr24")

# ----------------------------------------------------------------------
## GIAO DIỆN STREAMLIT CHÍNH
# ----------------------------------------------------------------------
st.set_page_config(page_title="Demo Softmax - Hybrid Detection", layout="wide")
st.title("🧠 Nhận diện trạng thái mất tập trung (Hybrid Detection)")

tab1, tab2 = st.tabs(["🔴 Dự đoán Live Camera", "🖼️ Dự đoán Ảnh Tĩnh"])
mesh_static = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

with tab1:
    st.warning("Phương pháp Hybrid: Dùng luật cứng (EAR < 0.20) cho BLINK, dùng Softmax cho các hành vi khác.")
    st.warning("Vui lòng chấp nhận yêu cầu truy cập camera từ trình duyệt của bạn.")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 4, 1]) 
    with col2: 
        webrtc_streamer(
            key="softmax_driver_live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_processor_factory=DrowsinessProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

with tab2:
    st.markdown("### Tải lên ảnh khuôn mặt để dự đoán trạng thái")
    uploaded_file = st.file_uploader("Chọn một ảnh khuôn mặt (.jpg, .png)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.info("Đang xử lý ảnh... ")
        
        # Xử lý và dự đoán
        result_img_rgb, predicted_label = process_static_image(uploaded_file, mesh_static, W, b, mean, std, id2label)
        
        st.markdown("---")
        
        col_img, col_res = st.columns([2, 1])
        
        with col_img:
            # Hiển thị ảnh đã được xử lý (đã là RGB)
            st.image(result_img_rgb, caption="Ảnh đã xử lý", use_column_width=True)
            
        with col_res:
            st.success("✅ Dự đoán Hoàn tất")
            st.metric(label="Trạng thái Dự đoán", value=predicted_label.upper())
            st.caption(f"Lưu ý: Delta EAR cho ảnh tĩnh luôn bằng 0.")

    else:
        st.info("Vui lòng tải lên một ảnh để bắt đầu dự đoán.")
