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

# Thêm khai báo mp_drawing (MP Solutions Drawing Utilities)
mp_drawing = mp.solutions.drawing_utils

# ======================================================================
# I. CẤU HÌNH VÀ HẰNG SỐ CHUNG
# ======================================================================

# --- Cấu hình chung ---
EPS = 1e-8 
NEW_WIDTH, NEW_HEIGHT = 640, 480 

# --- Cấu hình Drowsiness (Face Mesh) ---
MODEL_PATH = "softmax_model_best1.pkl"
SCALER_PATH = "scale1.pkl"
LABEL_MAP_PATH = "label_map_5cls.json"
SMOOTH_WINDOW = 5 
BLINK_THRESHOLD = 0.20 
N_FEATURES = 10 # Số lượng đặc trưng mong đợi

# --- Cấu hình Wheel (Hands) ---
WHEEL_MODEL_PATH = "softmax_wheel_model.pkl"
WHEEL_SCALER_PATH = "scaler_wheel.pkl"


# ======================================================================
# II. CÁC HÀM TÍNH TOÁN CƠ BẢN VÀ TẢI TÀI NGUYÊN
# ======================================================================

def softmax_predict(X, W, b):
    """Thực hiện dự đoán Softmax (Face Mesh)."""
    logits = X @ W + b
    return np.argmax(logits, axis=1)

def softmax_wheel(z):
    """Thực hiện dự đoán Softmax (Hands/Wheel)."""
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

@st.cache_resource
def get_mp_hands_instance():
    """Tạo instance MediaPipe Hands (dùng cho cache resource)."""
    return mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)

@st.cache_resource
def load_assets():
    """Tải tất cả tham số mô hình, scaler và label map."""
    W_WHEEL, b_WHEEL, X_mean_WHEEL, X_std_WHEEL, CLASS_NAMES_WHEEL = [None] * 5
    
    try:
        # --- 1. Tải Mô hình Face Mesh ---
        with open(MODEL_PATH, "rb") as f:
            model_data = joblib.load(f)
            W = model_data["W"]
            b = model_data["b"]
        with open(SCALER_PATH, "rb") as f:
            scaler_data = joblib.load(f)
            mean_data = scaler_data["X_mean"]
            std_data = scaler_data["X_std"]
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
            id2label = {int(v): k for k, v in label_map.items()}
            
        if W.shape[0] != N_FEATURES:
             st.error(f"LỖI KHÔNG TƯƠNG THÍCH: Mô hình FACE MESH yêu cầu {W.shape[0]} đặc trưng, nhưng ứng dụng này trích xuất {N_FEATURES} đặc trưng. Vui lòng kiểm tra lại file model!")
             st.stop()
             
        # --- 2. Tải Mô hình Wheel/Hands ---
        with open(WHEEL_MODEL_PATH, "rb") as f:
            wheel_model_data = joblib.load(f)
            W_WHEEL = wheel_model_data["W"]
            b_WHEEL = wheel_model_data["b"]
            CLASS_NAMES_WHEEL = wheel_model_data["classes"]
            
        with open(WHEEL_SCALER_PATH, "rb") as f:
            wheel_scaler_data = joblib.load(f)
            X_mean_WHEEL = wheel_scaler_data["X_mean"]
            X_std_WHEEL = wheel_scaler_data["X_std"]

        # ĐÃ SỬA: Bỏ X_std_WHEEL bị lặp thừa
        return W, b, mean_data, std_data, id2label, W_WHEEL, b_WHEEL, X_mean_WHEEL, X_std_WHEEL, CLASS_NAMES_WHEEL

    except FileNotFoundError as e:
        # Xử lý lỗi nếu thiếu file model nào đó
        st.error(f"LỖI FILE: Không tìm thấy file tài nguyên VÔ LĂNG hoặc KHUÔN MẶT. Vui lòng kiểm tra đường dẫn: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"LỖỖI LOAD DỮ LIỆU: Chi tiết: {e}")
        st.stop()

# Tải tài sản (Chạy một lần)
# ĐÃ SỬA: Bỏ X_std_WHEEL bị lặp thừa, chỉ gán 10 giá trị
W, b, mean, std, id2label, W_WHEEL, b_WHEEL, X_mean_WHEEL, X_std_WHEEL, CLASS_NAMES_WHEEL = load_assets()
classes = list(id2label.values())

# ======================================================================
# III. HÀM TRÍCH XUẤT ĐẶC TRƯNG KHUÔN MẶT (FACE MESH)
# ======================================================================

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

# ======================================================================
# IV. HÀM TRÍCH XUẤT ĐẶC TRƯNG VÔ LĂNG (WHEEL/HANDS)
# ======================================================================

def detect_wheel_circle(frame):
    """Sử dụng Hough Transform để phát hiện vô lăng."""
    # Frame phải là BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.0, minDist=120,
        param1=150, param2=40,
        minRadius=60, maxRadius=200
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]
        return (x, y, r)
    return None

def extract_wheel_features(image, hands_processor, wheel):
    """Trích xuất 128 đặc trưng tay và khoảng cách cổ tay chuẩn hóa."""
    if wheel is None: return None
    xw, yw, rw = wheel
    h, w, _ = image.shape
    feats_all = []

    # Image phải là RGB cho MediaPipe Hands
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = hands_processor.process(rgb)
    if not res.multi_hand_landmarks: return None

    for hand_landmarks in res.multi_hand_landmarks:
        feats = []
        for lm in hand_landmarks.landmark:
            feats.extend([lm.x, lm.y, lm.z])

        # Đặc trưng: Khoảng cách cổ tay chuẩn hóa
        hx = hand_landmarks.landmark[0].x * w
        hy = hand_landmarks.landmark[0].y * h
        dist = np.sqrt((xw - hx) ** 2 + (yw - hy) ** 2)
        feats.append(dist / rw)

        feats_all.extend(feats)

    # Đảm bảo đủ độ dài (128 = 64 * 2)
    feats_len_per_hand = 64
    expected_len = feats_len_per_hand * 2
    feats_all = feats_all[:expected_len]
    if len(feats_all) < expected_len:
        feats_all.extend([0.0] * (expected_len - len(feats_all)))

    return np.array(feats_all, dtype=np.float32)


# ======================================================================
# V. HÀM XỬ LÝ ẢNH TĨNH VÀ LIVE (Drowsiness)
# ======================================================================

def process_static_image(image_file, mesh, W, b, mean, std, id2label):
    # Đọc ảnh từ file uploader
    image = np.array(Image.open(image_file).convert('RGB'))
    
    # Resize ảnh để xử lý nhanh hơn và chuẩn hóa kích thước
    image_resized = cv2.resize(image, (NEW_WIDTH, NEW_HEIGHT))
    h, w = image_resized.shape[:2]
    
    # CHUẨN BỊ ẢNH CHO MEDIAPIPE (Bắt buộc lật để tính landmarks chính xác)
    image_for_mp = cv2.flip(image_resized, 1) # Lật ảnh trước khi xử lý
    
    # Xử lý MediaPipe
    results = mesh.process(image_for_mp)
    
    result_label = "Chưa tìm thấy khuôn mặt"
    
    # Chuẩn bị ảnh hiển thị: BGR và Lật lại để người dùng thấy ảnh đúng
    image_display_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    image_display_flipped = cv2.flip(image_display_bgr, 1)

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
        
        # 3. ÁP DỤNG LUẬT HEURISTIC CỨNG (Ưu tiên BLINK nếu mắt nhắm)
        if ear_avg < BLINK_THRESHOLD:
            result_label = "BLINK (Heuristic)"
        else:
            # 4. Chạy Softmax (10 đặc trưng)
            feats = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                              angle_pitch_extra, delta_ear_value, forehead_y, cheek_dist], dtype=np.float32)

            feats_scaled = (feats - mean[:N_FEATURES]) / (std[:N_FEATURES] + EPS)
            pred_idx = softmax_predict(np.expand_dims(feats_scaled, axis=0), W, b)[0]
            result_label = id2label.get(pred_idx, "UNKNOWN")
            
        # Hiển thị kết quả lên ảnh đã lật ngược lại (image_display_flipped)
        cv2.putText(image_display_flipped, f"Trang thai: {result_label.upper()}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        # Chuyển lại BGR sang RGB cho Streamlit
        final_image_rgb = cv2.cvtColor(image_display_flipped, cv2.COLOR_BGR2RGB)

        return final_image_rgb, result_label

    # Trường hợp không tìm thấy khuôn mặt
    cv2.putText(image_display_flipped, "KHONG TIM THAY KHUON MAT", (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    final_image_rgb = cv2.cvtColor(image_display_flipped, cv2.COLOR_BGR2RGB)
    
    return final_image_rgb, result_label

# ----------------------------------------------------------------------
## VI. HÀM XỬ LÝ ẢNH TĨNH (Wheel)
# ----------------------------------------------------------------------
def process_static_wheel_image(image_file, W_WHEEL, b_WHEEL, X_mean_WHEEL, X_std_WHEEL, CLASS_NAMES_WHEEL):
    # Đọc ảnh từ file uploader
    img_pil = Image.open(image_file).convert('RGB')
    img_np = np.array(img_pil)
    
    # Convert RGB to BGR for OpenCV processing (HoughCircles)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Khai báo mp_hands cục bộ
    mp_hands = mp.solutions.hands 
    hands_processor = get_mp_hands_instance()
    
    # 1. Phát hiện vô lăng
    wheel = detect_wheel_circle(img_bgr)
    
    if wheel is None:
        label = "KHÔNG TÌM THẤY VÔ LĂNG"
        cv2.putText(img_bgr, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), label

    # 2. Trích xuất đặc trưng
    features = extract_wheel_features(img_bgr.copy(), hands_processor, wheel)
    
    img_display = img_bgr # Bắt đầu vẽ trên ảnh BGR
    xw, yw, rw = wheel
    
    # Luôn vẽ vô lăng
    cv2.circle(img_display, (xw, yw), rw, (0, 255, 0), 2)
    cv2.circle(img_display, (xw, yw), 5, (0, 0, 255), -1)

    if features is None:
        label = "OFF-WHEEL (Tay không được phát hiện)"
        color = (0, 0, 255) # Đỏ
        cv2.putText(img_display, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), "OFF-WHEEL" # Trả về nhãn Off-wheel


    # 3. Chuẩn hóa và dự đoán
    X_sample = features.reshape(1, -1)
    X_scaled = (X_sample - X_mean_WHEEL) / (X_std_WHEEL + EPS) # Sử dụng X_mean_WHEEL, X_std_WHEEL

    z = X_scaled @ W_WHEEL + b_WHEEL # Sử dụng W_WHEEL, b_WHEEL
    probabilities = softmax_wheel(z)[0]

    predicted_index = np.argmax(probabilities)
    predicted_class = CLASS_NAMES_WHEEL[predicted_index]
    confidence = probabilities[predicted_index] * 100
    
    # --- Visualization (Tay) ---
    rgb_for_drawing = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    res_for_drawing = hands_processor.process(rgb_for_drawing)
    
    if res_for_drawing.multi_hand_landmarks:
        for hand_landmarks in res_for_drawing.multi_hand_landmarks:
            # SỬA LỖI: Thay thế mp_hands.drawing_utils bằng mp_drawing
            mp_drawing.draw_landmarks( 
                img_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị nhãn dự đoán
    text = f"{predicted_class.upper()} ({confidence:.1f}%)"
    color = (0, 0, 255) if predicted_class == "off-wheel" else (0, 255, 0)
    
    # Căn chỉnh text để không trùng với vô lăng
    cv2.putText(img_display, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    return cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), predicted_class.upper()


# ======================================================================
# VII. LỚP XỬ LÝ VIDEO LIVE (WEBRTC PROCESSOR)
# ======================================================================
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

                feats_scaled = (feats - self.mean[:self.N_FEATURES]) / (self.std[:N_FEATURES] + EPS)
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

# ======================================================================
# VIII. GIAO DIỆN STREAMLIT CHÍNH
# ======================================================================
st.set_page_config(page_title="Demo nhận diện các hành vi mất tập trung - Softmax ", layout="wide")

tab1, tab2, tab3 = st.tabs(["🔴 Dự đoán Live Camera", "🖼️ Dự đoán Ảnh Tĩnh (Khuôn Mặt)", "🚗 Kiểm tra Vô Lăng (Tay)"])
mesh_static = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

with tab1:
    st.header("1. Nhận diện Trạng thái Khuôn mặt (Live Camera)")
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
    st.header("2. Dự đoán Ảnh Tĩnh (Khuôn Mặt)")
    st.markdown("### Tải lên ảnh khuôn mặt để dự đoán trạng thái (Ngủ gật/Mất tập trung)")
    uploaded_file = st.file_uploader("Chọn một ảnh khuôn mặt (.jpg, .png)", type=["jpg", "png", "jpeg"], key="face_upload")

    if uploaded_file is not None:
        st.info("Đang xử lý ảnh... ")
        
        result_img_rgb, predicted_label = process_static_image(uploaded_file, mesh_static, W, b, mean, std, id2label)
        
        st.markdown("---")
        
        col_img, col_res = st.columns([2, 1])
        
        with col_img:
            st.image(result_img_rgb, caption="Ảnh đã xử lý", use_column_width=True)
            
        with col_res:
            st.success("✅ Dự đoán Hoàn tất")
            st.metric(label="Trạng thái Dự đoán", value=predicted_label.upper())
            st.caption(f"Lưu ý: Delta EAR cho ảnh tĩnh luôn bằng 0.")

    else:
        st.info("Vui lòng tải lên một ảnh để bắt đầu dự đoán.")

with tab3:
    st.header("3. Kiểm tra Vị trí Tay (Vô Lăng)")
    st.warning(f"Mô hình Vô Lăng nhận diện: {CLASS_NAMES_WHEEL}")
    st.markdown("### Tải lên ảnh tay trên/rời vô lăng để dự đoán")
    uploaded_wheel_file = st.file_uploader("Chọn một ảnh vô lăng (.jpg, .png)", type=["jpg", "png", "jpeg"], key="wheel_upload")
    
    if uploaded_wheel_file is not None:
        st.info("Đang xử lý ảnh...")
        
        # Xử lý và dự đoán
        result_img_rgb, predicted_label = process_static_wheel_image(uploaded_wheel_file, W_WHEEL, b_WHEEL, X_mean_WHEEL, X_std_WHEEL, CLASS_NAMES_WHEEL)
        
        st.markdown("---")
        
        col_img, col_res = st.columns([2, 1])

        with col_img:
            st.image(result_img_rgb, caption="Ảnh đã xử lý (Vô lăng, Tay)", use_column_width=True)
            
        with col_res:
            st.success("✅ Dự đoán Hoàn tất")
            st.metric(label="Vị trí Tay Dự đoán", value=predicted_label.upper())
            st.caption("Kiểm tra màu sắc: Xanh lá (On-wheel), Đỏ (Off-wheel)")
            
    else:
        st.info("Vui lòng tải lên một ảnh lái xe để kiểm tra vị trí tay.")

