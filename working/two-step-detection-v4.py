# 아에 안 움직이는 객체의 경우에도 object가 움직인다고 하는 문제를 해결하기 위한 코드
# 너무 민감함

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import deque

# GPU 사용 가능 여부 확인 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 두 개의 YOLO 모델 로드
furniture_model = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/original/yolov8m.pt')
person_wheelchair_model = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/4-1/best.pt')

# 안전한 가구 클래스 ID (COCO 데이터셋 기준)
safe_furniture_classes = [56, 57, 59]  # 56: chair, 57: couch, 59: bed

# 클래스 이름 매핑
class_names = {56: 'chair', 57: 'couch', 59: 'bed'}

# 전역 변수로 가구 정보 저장
furniture_info = []


# 객체 추적을 위한 클래스
class TrackedObject:
    def __init__(self, box, center):
        self.box = box
        self.center = center
        self.positions = deque(maxlen=30)  # 1초(30프레임) 동안의 위치 저장
        self.positions.append(center)
        self.stationary_count = 0
        self.moving_count = 0
        self.state = "Initializing"  # 초기 상태
        self.near_safe_object = False


# 객체 추적 정보
tracked_objects = {}

# 상수 정의
MOVEMENT_THRESHOLD = 5  # 픽셀 단위
STATIONARY_THRESHOLD = 90  # 3초 (30fps 기준)
MOVING_THRESHOLD = 30  # 1초 (30fps 기준)
SAFE_DISTANCE = 100  # 안전 객체와의 거리 (픽셀 단위)

MOVEMENT_THRESHOLD = 3  # 더 작은 움직임을 감지하도록 조정
STATIONARY_THRESHOLD = 45  # 1.5초 (30fps 기준)
MOVING_THRESHOLD = 15  # 0.5초 (30fps 기준)
SAFE_DISTANCE = 150  # 안전 객체와의 거리를 더 크게 설정 (픽셀 단위)



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


def calculate_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def initialize_furniture(first_frame):
    global furniture_info
    furniture_results = furniture_model(first_frame)
    for r in furniture_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            if cls in safe_furniture_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                furniture_info.append({
                    'box': (x1, y1, x2, y2),
                    'class': cls,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })
    print(f"가구 감지 완료: {len(furniture_info)}개의 안전한 가구 감지됨")


def update_object_state(obj):
    if len(obj.positions) < 2:
        return "Initializing"

    recent_movement = calculate_distance(obj.positions[-1], obj.positions[-2])

    if recent_movement < MOVEMENT_THRESHOLD:
        obj.stationary_count += 1
        obj.moving_count = 0
    else:
        obj.moving_count += 1
        obj.stationary_count = 0

    if obj.stationary_count > STATIONARY_THRESHOLD:
        return "Stationary"
    elif obj.moving_count > MOVING_THRESHOLD:
        return "Moving"
    else:
        return obj.state  # 상태 유지


def is_near_safe_object(obj):
    for furniture in furniture_info:
        if calculate_distance(obj.center, furniture['center']) < SAFE_DISTANCE:
            return True
    return False


def process_frame(frame, frame_count):
    global tracked_objects

    detected_objects = {'person': [], 'wheelchair': []}

    # 사람과 휠체어 감지
    person_wheelchair_results = person_wheelchair_model(frame)
    for r in person_wheelchair_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if cls == 0:  # person
                detected_objects['person'].append({
                    'box': (x1, y1, x2, y2),
                    'center': center
                })
            elif cls == 1:  # wheelchair
                detected_objects['wheelchair'].append({
                    'box': (x1, y1, x2, y2),
                    'center': center
                })

    # 객체 추적 및 상태 업데이트
    current_objects = {}
    for i, obj in enumerate(detected_objects['person']):
        if i in tracked_objects:
            tracked_obj = tracked_objects[i]
            tracked_obj.box = obj['box']
            tracked_obj.center = obj['center']
            tracked_obj.positions.append(obj['center'])
        else:
            tracked_obj = TrackedObject(obj['box'], obj['center'])

        tracked_obj.state = update_object_state(tracked_obj)
        tracked_obj.near_safe_object = is_near_safe_object(tracked_obj)
        current_objects[i] = tracked_obj

    tracked_objects = current_objects

    # 화면에 정보 표시
    for i, obj in tracked_objects.items():
        x1, y1, x2, y2 = obj.box
        if obj.near_safe_object and obj.state != "Moving":
            color = (0, 255, 0)  # 녹색 (안전)
            status = f"Safe ({obj.state})"
        elif obj.state == "Moving":
            color = (0, 255, 255)  # 노란색 (움직이는 중)
            status = "Moving"
        else:
            color = (0, 0, 255)  # 빨간색 (위험)
            status = f"Unsafe ({obj.state})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Person {i}: {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 가구 정보 표시
    for furniture in furniture_info:
        x1, y1, x2, y2 = furniture['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Safe: {class_names[furniture['class']]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    cv2.putText(frame, f"Frame: {frame_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


# 메인 실행 부분
def main():
    input_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/original/test-video.mp4'
    output_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/output/v4-output.mp4'

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 첫 프레임으로 가구 정보 초기화
    ret, first_frame = cap.read()
    if ret:
        initialize_furniture(first_frame)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        processed_frame = process_frame(frame, frame_count)
        out.write(processed_frame)

        cv2.imshow('Processing Video', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("비디오 처리 완료")


if __name__ == "__main__":
    main()