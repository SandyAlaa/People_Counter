import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
from collections import deque
import cv2
import av

# Define STUN server to handle WebRTC NAT traversal
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_yolo_model():
    # Cache the model so it only loads once when the app starts
    return YOLO("yolov8n.pt")

class PeopleCounterProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_yolo_model()
        self.conf_threshold = 0.4
        self.person_class_id = 0
        self.recent_counts = deque(maxlen=30)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Processes each video frame from the browser's webcam."""
        # Convert incoming WebRTC frame to OpenCV BGR format
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 inference
        results = self.model(img, verbose=False)[0]

        current_count = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == self.person_class_id and conf >= self.conf_threshold:
                current_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.recent_counts.append(current_count)
        avg_count = round(sum(self.recent_counts) / len(self.recent_counts))

        # Overlay text on frame
        cv2.putText(
            img, f"Current Frame: {current_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.putText(
            img, f"Averaged ({len(self.recent_counts)}f): {avg_count}", (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2
        )

        # Convert modified image back to WebRTC frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit Dashboard UI ---
st.set_page_config(page_title="AI People Counter", layout="centered")
st.title("📹 Live Web People Counter")
st.write("Real-time webcam people detection powered by YOLOv8 and Streamlit.")

# Streamlit-WebRTC Streamer Component
webrtc_streamer(
    key="people-counter",
    video_processor_factory=PeopleCounterProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)