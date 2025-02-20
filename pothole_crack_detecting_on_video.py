import cv2
from ultralytics import YOLO
import numpy as np

# 모델 로드
pothole_model = YOLO(r"D:\pothole_best.pt")  # 포트홀 탐지 모델
crack_model = YOLO(r"I:\이채영\models\crack_best2.pt")  # 크랙 탐지 모델

# 비디오 파일 경로 설정
video_path = r"D:\pothole_detectiong\vecteezy_summer-road-trip-through-the-english-countryside_30223185.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오가 정상적으로 열렸는지 확인
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video reading completed or failed.")
        break
    
    if out is None:
        height, width, _ = frame.shape
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(r'D:\pothole_detectiong\pothole_crack_detected.mp4', fourcc, fps, (width, height))
        if not out.isOpened():
            print("Error: Cannot open output video file.")
            cap.release()
            exit()

    # ----------------- 포트홀 모델 예측 -----------------
    pothole_results = pothole_model(frame)
    pothole_masks = pothole_results[0].masks
    pothole_boxes = pothole_results[0].boxes
    
    if pothole_masks is not None:
        mask_data = pothole_masks.data.cpu().numpy()
        for i, mask in enumerate(mask_data):
            mask_resized = cv2.resize(mask, (width, height))
            mask_colored = np.zeros_like(frame, dtype=np.uint8)
            mask_colored[mask_resized > 0.5] = [0, 0, 255]  # 🔴 빨간색: 포트홀
            frame = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)

            if i < len(pothole_boxes):
                box = pothole_boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = f"pothole: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ----------------- 크랙 모델 예측 -----------------
    crack_results = crack_model(frame)
    crack_masks = crack_results[0].masks
    crack_boxes = crack_results[0].boxes
    
    if crack_masks is not None:
        mask_data = crack_masks.data.cpu().numpy()
        for i, mask in enumerate(mask_data):
            mask_resized = cv2.resize(mask, (width, height))
            mask_colored = np.zeros_like(frame, dtype=np.uint8)
            mask_colored[mask_resized > 0.5] = [0, 255, 0]  # 🟢 초록색: 크랙
            frame = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)

            if i < len(crack_boxes):
                box = crack_boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = f"crack: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 처리된 프레임 저장
    if out is not None:
        out.write(frame)

# 비디오 캡처 및 저장 객체 해제
cap.release()
if out is not None:
    out.release()

print("Video processing with pothole and crack detection completed successfully.")
