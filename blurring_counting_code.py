import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8m.pt')

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
        out = cv2.VideoWriter('output_blurred_with_count.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # YOLOv8로 예측 수행
    results = model(frame)

    # 카운팅을 위한 변수 초기화
    vehicle_count = 0
    person_count = 0
    vehicle_types = {
        'car': 0,
        'motorbike': 0,
        'bus': 0,
        'truck': 0
    }

    # 예측 결과 처리
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if conf > 0.5:  # 신뢰도 기준 설정
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls]

                # 차량 카운팅 및 색상 설정
                if label in vehicle_types:
                    vehicle_types[label] += 1
                    vehicle_count += 1
                    color = color_map.get(label, (255, 255, 255))  # 해당 차량에 맞는 색상

                    # 블러링 적용 (차량 영역)
                    sub_img = frame[y1:y2, x1:x2]
                    blur = cv2.GaussianBlur(sub_img, (23, 23), 30)
                    frame[y1:y2, x1:x2] = blur

                    # 바운딩 박스와 라벨 표시 (객체 클래스 글씨 크기 조금 줄임)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 사람 카운팅 및 블러링 적용
                elif label == 'person':
                    person_count += 1
                    sub_img = frame[y1:y2, x1:x2]
                    blur = cv2.GaussianBlur(sub_img, (23, 23), 30)
                    frame[y1:y2, x1:x2] = blur

                    # 바운딩 박스와 라벨 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 카운팅 정보 표시
    count_text = f'Vehicles: {vehicle_count}, Persons: {person_count}'
    vehicle_type_text = ', '.join([f'{k.capitalize()}: {v}' for k, v in vehicle_types.items()])

    # 텍스트 크기 계산
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.8
    text_thickness = 2
    text_padding = 10

    # 첫 번째 텍스트 크기 계산 (카운팅 정보)
    count_text_size, count_baseline = cv2.getTextSize(count_text, text_font, text_scale, text_thickness)
    
    # 두 번째 텍스트 크기 계산 (차량 종류별 카운트)
    vehicle_type_text_size, vehicle_type_baseline = cv2.getTextSize(vehicle_type_text, text_font, text_scale, text_thickness)

    # 배경 박스 크기 결정 (가장 긴 텍스트의 너비와 높이에 패딩 추가)
    box_width = max(count_text_size[0], vehicle_type_text_size[0]) + text_padding * 2
    box_height = count_text_size[1] + vehicle_type_text_size[1] + text_padding * 3 + count_baseline + vehicle_type_baseline

    # 투명한 배경을 위한 박스 그리기
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_width, 5 + box_height), (0, 0, 0), -1)  # 검은색 배경
    alpha = 0.5  # 투명도 설정 (0.0 완전 투명, 1.0 완전 불투명)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 카운팅 정보 텍스트 표시 (글씨 크기 줄임)
    cv2.putText(frame, count_text, (10, 30 + text_padding), text_font, text_scale, (255, 255, 255), text_thickness)
    cv2.putText(frame, vehicle_type_text, (10, 70 + text_padding), text_font, text_scale, (255, 255, 255), text_thickness)

    # 처리된 프레임을 비디오 파일로 저장
    out.write(frame)

cap.release()
out.release()


