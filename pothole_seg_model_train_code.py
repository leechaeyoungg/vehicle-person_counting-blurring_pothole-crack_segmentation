from ultralytics import YOLO

def train_model():
    # YOLOv8m 모델 로드 (세그멘테이션용)
    model = YOLO('yolov8m-seg.pt')  # pre-trained 세그멘테이션 모델 사용

    # 데이터셋 경로 설정
    data_yaml = r"D:\Pothole Segmentation.v14i.yolov8\data.yaml"

    # 모델 훈련
    model.train(data=data_yaml, epochs=100, imgsz=640, batch=16, name='pothole_segmentation_yolov8m')

    # 모델 검증 (선택 사항, 훈련 후 모델의 성능을 평가)
    model.val(data=data_yaml)

    # 훈련된 모델 저장
    model.save(r'D:\pothole_segmentation_yolov8m.pt')

if __name__ == '__main__':
    train_model()

