import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
vehicle_person_model = YOLO('yolov8m.pt')
crack_model = YOLO(r"D:\crack_detection_yolov8m.pt")

# 비디오 파일 경로 설정
video_path = r"D:\2023_08_09_13_15_25.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오 저장을 위한 설정 (원본과 동일한 프레임 크기 및 FPS 사용)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

# 색상 설정 (차량 종류별로 다른 색상)
color_map = {
    'car': (0, 255, 0),       # 초록색
    'motorbike': (255, 0, 0), # 파란색
    'bus': (0, 0, 255),       # 빨간색
    'truck': (255, 255, 0)    # 노란색
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if out is None:
        height, width, _ = frame.shape
        out = cv2.VideoWriter(r'D:\2023_08_09_13_15_25_combined.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # 차량 및 사람 감지를 위한 YOLOv8 예측 수행
    vehicle_person_results = vehicle_person_model(frame)

    vehicle_count = 0
    person_count = 0
    vehicle_types = {
        'car': 0,
        'motorbike': 0,
        'bus': 0,
        'truck': 0
    }

    for result in vehicle_person_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if conf > 0.5:  # 신뢰도 기준 설정
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = vehicle_person_model.names[cls]

                # 차량 종류에 따라 카운트 및 블러 처리
                if label in vehicle_types:
                    vehicle_types[label] += 1
                    vehicle_count += 1
                    sub_img = frame[y1:y2, x1:x2]
                    blur = cv2.GaussianBlur(sub_img, (23, 23), 30)
                    frame[y1:y2, x1:x2] = blur

                    # 바운딩 박스와 라벨 표시 (차량 종류에 따라 색상 다르게 설정)
                    color = color_map.get(label, (255, 255, 255))  # 기본적으로 흰색으로 설정
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                elif label == 'person':
                    person_count += 1
                    sub_img = frame[y1:y2, x1:x2]
                    blur = cv2.GaussianBlur(sub_img, (23, 23), 30)
                    frame[y1:y2, x1:x2] = blur

                    # 바운딩 박스와 라벨 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 차량 종류별 카운트 및 사람 카운트 표시
    count_text = f'Vehicles: {vehicle_count}, Persons: {person_count}'
    vehicle_type_text = ', '.join([f'{k.capitalize()}: {v}' for k, v in vehicle_types.items()])
    cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, vehicle_type_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 크랙 감지를 위한 YOLOv8 예측 수행
    crack_results = crack_model(frame)

    # 세그멘테이션 마스크와 라벨, 정확도 표시
    masks = crack_results[0].masks  # 세그멘테이션 마스크 가져오기
    boxes = crack_results[0].boxes  # 바운딩 박스 및 라벨 정보 가져오기
    
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
            label = f"{crack_model.names[class_id]} {confidence:.2f}"

            # 라벨과 정확도 표시
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 처리된 프레임을 비디오 파일로 저장
    out.write(frame)

cap.release()
out.release()
