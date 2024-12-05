# model_segmentation = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/segment/yolo11x-seg.pt')  # segmentation 모델 경로
# model_detection = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/original/yolov8m.pt')
# model_detection = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/4-1/best.pt')
# input_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/original/men-sofa.mp4'
# output_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/output/proto-v6.mp4'

# 1. 안전가구 segmentation으로 탐지


import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import deque

# GPU 사용 가능 여부 확인 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 모델 로드
# 안전 가구 탐지를 위한 세그멘테이션 모델
model_segmentation = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/segment/yolo11x-seg.pt')  # segmentation 모델 경로
model_detection = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/original/yolov8m.pt') # 중간 모델
model_detection = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/original/yolov8n.pt') # 작은 모델
# model_detection= YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/4-1/best.pt') # trained model

# 안전한 가구 클래스 ID (COCO 데이터셋 기준)
safe_furniture_classes = [56, 57, 59]  # 56: chair, 57: couch, 59: bed

# 클래스 이름 매핑
class_names = {56: 'chair', 57: 'couch', 59: 'bed'}

# 전역 변수로 가구 정보 저장
furniture_info = []


# 객체 추적을 위한 클래스
class TrackedObject:
    def __init__(self, box, center, track_id):
        self.box = box
        self.center = center
        self.track_id = track_id
        self.positions = deque(maxlen=30)
        self.positions.append(center)
        self.stationary_count = 0
        self.state = "Initializing"
        self.near_safe_object = False


# 전역 변수들
tracked_objects = {}

# 상수 정의 (fps는 이후에 설정)
MOVEMENT_THRESHOLD = 2.5  # 작은 움직임도 감지하도록 임계값 낮춤
STATIONARY_THRESHOLD = None  # fps 기반으로 설정할 예정


def calculate_iou(box1, box2):
    # 박스 좌표: (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.


def initialize_furniture(first_frame):
    global furniture_info
    # 안전 가구 탐지를 위해 세그멘테이션 모델 사용
    furniture_results = model_segmentation(first_frame, classes=safe_furniture_classes)
    for r in furniture_results:
        masks = r.masks  # 세그멘테이션 마스크
        boxes = r.boxes
        for i, box in enumerate(boxes):
            cls = int(box.cls)
            if cls in safe_furniture_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                mask = masks.data[i].cpu().numpy()  # 마스크 데이터
                furniture_info.append({
                    'box': (x1, y1, x2, y2),
                    'class': cls,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'mask': mask
                })
    print(f"가구 검지 완료: {len(furniture_info)}개의 안전한 가구 검지됨")


def update_object_state(obj):
    if len(obj.positions) < 2:
        return obj.state  # 위치 정보가 충분하지 않으면 상태 유지

    recent_movement = calculate_distance(obj.positions[-1], obj.positions[-2])

    if recent_movement >= MOVEMENT_THRESHOLD:
        # 움직임이 감지되면 즉시 moving 상태로 전환
        obj.stationary_count = 0
        obj.state = "Moving"
    else:
        obj.stationary_count += 1
        if obj.stationary_count > STATIONARY_THRESHOLD:
            obj.state = "Stationary"
            print(f"알람 : Object ID {obj.track_id} 위험을 감지 했습니다.")

    return obj.state


def is_near_safe_object(obj):
    for furniture in furniture_info:
        iou = calculate_iou(obj.box, furniture['box'])
        if iou > 0:
            return True
    return False


def calculate_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def process_frame(frame, frame_count):
    global tracked_objects

    # 사람 감지 및 추적
    results = model_detection.track(frame, persist=True, classes=[0])  # 클래스 0: person

    # 현재 프레임의 객체들
    current_tracked_objects = {}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.id is None:
                continue  # 추적 ID가 없는 경우 스킵
            track_id = int(box.id)
            cls = int(box.cls)
            if cls != 0:
                continue  # person 클래스가 아니면 스킵

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # 트랙 ID를 사용하여 객체 상태 관리
            if track_id in tracked_objects:
                tracked_obj = tracked_objects[track_id]
                tracked_obj.box = (x1, y1, x2, y2)
                tracked_obj.center = center
                tracked_obj.positions.append(center)
            else:
                tracked_obj = TrackedObject((x1, y1, x2, y2), center, track_id)

            # 상태 업데이트
            tracked_obj.state = update_object_state(tracked_obj)
            tracked_obj.near_safe_object = is_near_safe_object(tracked_obj)
            current_tracked_objects[track_id] = tracked_obj

            # 화면에 정보 표시
            if tracked_obj.near_safe_object:
                if tracked_obj.state == "Stationary":
                    color = (0, 255, 0)  # 녹색 (안전)
                    status = "Safe"
                elif tracked_obj.state == "Moving":
                    color = (0, 255, 255)  # 노란색 (안전 객체 근처에서 움직임)
                    status = "Moving Near Safe Object"
                elif tracked_obj.state == "Initializing":
                    color = (255, 0, 0)  # 파란색
                    status = "Initializing Near Safe Object"
                else:
                    color = (0, 255, 255)  # 노란색
                    status = f"{tracked_obj.state} Near Safe Object"
            else:
                if tracked_obj.state == "Stationary":
                    color = (0, 0, 255)  # 빨간색 (위험)
                    status = "Unsafe (Stationary)"
                elif tracked_obj.state == "Moving":
                    color = (0, 255, 255)  # 노란색 (움직이는 중)
                    status = "Moving"
                elif tracked_obj.state == "Initializing":
                    color = (255, 0, 0)  # 파란색
                    status = "Initializing"
                else:
                    color = (0, 0, 255)  # 빨간색
                    status = f"Unsafe ({tracked_obj.state})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Person {track_id}: {status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 이전 프레임의 객체 정보를 현재 프레임의 객체로 업데이트
    tracked_objects = current_tracked_objects

    # 가구 정보 표시 (세그멘테이션 마스크 사용)
    for furniture in furniture_info:
        mask = furniture['mask']

        # 마스크의 크기를 프레임 크기에 맞게 조정
        resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 마스크를 적용하여 안전 가구 영역을 시각화
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        color_mask[resized_mask == 1] = (0, 255, 0)  # 녹색 마스크
        alpha = 0.5  # 투명도 설정
        frame = cv2.addWeighted(frame, 1, color_mask, alpha, 0)

        x1, y1, x2, y2 = furniture['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Safe: {class_names[furniture['class']]}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Frame: {frame_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


# 메인 실행 부분
def main():
    global STATIONARY_THRESHOLD

    # 입력 및 출력 비디오 경로를 실제 파일 위치로 수정하세요
    input_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/original/cafe.mp4'
    output_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/output/cafe-output-v6.mp4'

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 프레임률 기반으로 임계값 설정
    STATIONARY_THRESHOLD = fps * 1  # 1초 동안 정지하면 'Stationary'로 판단

    # 첫 프레임으로 가구 정보 초기화
    ret, first_frame = cap.read()
    if ret:
        initialize_furniture(first_frame)
    else:
        print("첫 프레임을 읽을 수 없습니다.")
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        processed_frame = process_frame(frame, frame_count)
        out.write(processed_frame)

        # 화면에 결과를 표시하려면 아래 주석을 해제하세요.
        # cv2.imshow('Processing Video', processed_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("비디오 처리 완료")


if __name__ == "__main__":
    main()

