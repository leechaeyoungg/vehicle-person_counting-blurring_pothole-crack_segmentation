import cv2
from ultralytics import YOLO
import numpy as np

# 크랙 감지 모델 로드
model = YOLO(r"D:\crack_best2.pt")

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
        out = cv2.VideoWriter('output_crack_segmentation2.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # 모델로 예측 수행
    results = model(frame)

    # 세그멘테이션 마스크와 라벨, 정확도 표시
    masks = results[0].masks  # 세그멘테이션 마스크 가져오기
    boxes = results[0].boxes  # 바운딩 박스 및 라벨 정보 가져오기
    
    if masks is not None:
        mask_data = masks.data.cpu().numpy()  # 마스크 데이터를 NumPy 배열로 변환
        for i, mask in enumerate(mask_data):
            mask_resized = cv2.resize(mask, (width, height))  # 마스크 크기를 프레임 크기에 맞게 조정
            mask_colored = np.zeros_like(frame, dtype=np.uint8)  # 프레임 크기에 맞게 빈 컬러 마스크 생성
            mask_colored[mask_resized > 0.5] = [0, 0, 255]  # 빨간색으로 마스크 영역 설정
            frame = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)  # 마스크와 원본 프레임 합성

            # 바운딩 박스의 좌표를 얻어와 라벨과 정확도 표시
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = f"crack {confidence:.2f}"

            # 라벨과 정확도 표시 (클래스 이름과 정확도만)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 처리된 프레임을 비디오 파일로 저장
    out.write(frame)

cap.release()
out.release()
