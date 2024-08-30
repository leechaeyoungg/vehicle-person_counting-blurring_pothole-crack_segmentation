import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드 (YOLOv8m)
model = YOLO('yolov8m.pt')

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
        # 저장할 비디오 파일 설정
        out = cv2.VideoWriter('output_blurred_with_labels.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # YOLOv8로 예측 수행
    results = model(frame)

    vehicle_count = 0
    person_count = 0
    vehicle_types = {
        'car': 0,
        'motorbike': 0,
        'bus': 0,
        'truck': 0
    }

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if conf > 0.5:  # 신뢰도 기준 설정
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls]

                # 차량 종류에 따라 카운트 및 블러 처리
                if label in vehicle_types:
                    vehicle_types[label] += 1
                    vehicle_count += 1
                    sub_img = frame[y1:y2, x1:x2]
                    blur = cv2.GaussianBlur(sub_img, (23, 23), 30)
                    frame[y1:y2, x1:x2] = blur

                    # 바운딩 박스와 라벨 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                elif label == 'person':
                    person_count += 1
                    sub_img = frame[y1:y2, x1:x2]
                    blur = cv2.GaussianBlur(sub_img, (23, 23), 30)
                    frame[y1:y2, x1:x2] = blur

                    # 바운딩 박스와 라벨 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 차량 종류별 카운트 및 사람 카운트 표시
    count_text = f'Vehicles: {vehicle_count}, Persons: {person_count}'
    vehicle_type_text = ', '.join([f'{k.capitalize()}: {v}' for k, v in vehicle_types.items()])
    cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, vehicle_type_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 처리된 프레임을 비디오 파일로 저장
    out.write(frame)

cap.release()
out.release()
