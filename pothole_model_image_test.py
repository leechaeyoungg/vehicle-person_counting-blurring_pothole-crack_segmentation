import cv2
from ultralytics import YOLO
import numpy as np

# YOLO 모델 로드
model = YOLO(r"D:\pothole_best.pt")

# 이미지 로드
image_path = r"D:\9058_47434_583.jpg"
image = cv2.imread(image_path)

# 모델로 예측 수행
results = model(image)

# 결과 이미지 생성
annotated_image = image.copy()

# 원본 이미지 크기
image_height, image_width = image.shape[:2]

# 세그멘테이션 결과 적용
for mask in results[0].masks.data:
    # 마스크를 원본 이미지 크기로 리사이즈
    mask_resized = cv2.resize(mask.cpu().numpy(), (image_width, image_height))

    # 마스크를 이미지에 적용
    annotated_image[mask_resized > 0.5] = (0, 255, 0)  # 세그멘테이션 마스크를 녹색으로 표시

# 클래스 이름과 정확도 추가
for box in results[0].boxes:
    cls_id = int(box.cls[0])  # cls 속성을 올바르게 참조
    confidence = float(box.conf[0])
    
    # 모델의 names 속성에서 클래스 이름을 가져옴
    label = f"{model.names[cls_id]} {confidence:.2f}"

    # 바운딩 박스 좌표 추출
    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

    # 이미지에 텍스트로 클래스 이름과 정확도 추가
    cv2.putText(annotated_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 결과 이미지 저장
output_path = r"D:\MicrosoftTeams-image_32.jpg"
cv2.imwrite(output_path, annotated_image)


