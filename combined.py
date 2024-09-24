import cv2
from ultralytics import YOLO
import numpy as np

# 포트홀 및 크랙 감지 모델 로드
pothole_model = YOLO(r"D:\pothole_best.pt")
crack_model = YOLO(r"D:\crack_best2.pt")

# 비디오 파일 경로 설정
video_path = r"D:\2023_08_09_13_15_25.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오 저장을 위한 설정 (원본과 동일한 프레임 크기 및 FPS 사용)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if out is None:
        height, width, _ = frame.shape
        out = cv2.VideoWriter('output_pothole_crack_segmentation2.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # 포트홀 감지 수행
    pothole_results = pothole_model(frame)

    # 크랙 감지 수행
    crack_results = crack_model(frame)

    # 포트홀 세그멘테이션 마스크 및 라벨 표시
    if pothole_results[0].masks is not None:
        pothole_masks = pothole_results[0].masks.data.cpu().numpy()  # 세그멘테이션 마스크 가져오기
        pothole_boxes = pothole_results[0].boxes
        for i, mask in enumerate(pothole_masks):
            mask_resized = cv2.resize(mask, (width, height))  # 마스크 크기 조정
            mask_colored = np.zeros_like(frame, dtype=np.uint8)
            mask_colored[mask_resized > 0.5] = [0, 255, 0]  # 초록색으로 마스크 영역 설정
            frame = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)

            # 포트홀 라벨 표시
            box = pothole_boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = f"pothole {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 크랙 세그멘테이션 마스크 및 라벨 표시
    if crack_results[0].masks is not None:
        crack_masks = crack_results[0].masks.data.cpu().numpy()  # 세그멘테이션 마스크 가져오기
        crack_boxes = crack_results[0].boxes
        for i, mask in enumerate(crack_masks):
            mask_resized = cv2.resize(mask, (width, height))  # 마스크 크기 조정
            mask_colored = np.zeros_like(frame, dtype=np.uint8)
            mask_colored[mask_resized > 0.5] = [0, 0, 255]  # 빨간색으로 마스크 영역 설정
            frame = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)

            # 크랙 라벨 표시
            box = crack_boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = f"crack {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 처리된 프레임을 비디오 파일로 저장
    out.write(frame)

cap.release()
out.release()
