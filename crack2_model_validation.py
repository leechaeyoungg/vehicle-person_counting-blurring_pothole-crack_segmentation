from ultralytics import YOLO

def validate_model():
    # 모델 로드
    model = YOLO('D:\crack_best2.pt')

    # 데이터셋 경로 설정
    data_yaml = r'D:\crack-seg.v2-withoutaug.yolov8\data.yaml'  # 경로 수정

    # 모델 검증
    model.val(data=data_yaml)

if __name__ == '__main__':
    validate_model()


