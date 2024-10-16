import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import deque

# GPU 사용 가능 여부 확인 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 두 개의 YOLO 모델 로드
furniture_model = YOLO('/trained-models/original/yolov8m.pt')
person_wheelchair_model = YOLO('/trained-models/4-1/best.pt')

# 안전한 가구 클래스 ID (COCO 데이터셋 기준)
safe_furniture_classes = [56, 57, 59]  # 56: chair, 57: couch, 59: bed

# 클래스 이름 매핑
class_names = {56: 'chair', 57: 'couch', 59: 'bed'}

# 전역 변수로 가구 정보 저장
furniture_info = []

# 사람의 이전 위치를 저장할 딕셔너리
previous_positions = {}

# 객체 추적을 위한 전역 변수
last_object_positions = {}
object_id_counter = 0

# 설정값
MOVEMENT_THRESHOLD = 5  # 움직임 감지를 위한 임계값 (픽셀 단위)
STATIONARY_THRESHOLD = 30 * 1  # 1초 (30fps 기준)
UNSAFE_THRESHOLD = 30 * 2  # 2초 (30fps 기준)
POSITION_HISTORY_LENGTH = 10  # 위치 히스토리 길이
SIZE_CHANGE_THRESHOLD = 0.05  # 크기 변화 임계값 (5%)
UNSAFE_MAINTAIN_THRESHOLD = 30 * 1  # Unsafe 상태 유지 시간 (1초)
WHEELCHAIR_IOU_THRESHOLD = 0.3  # 휠체어와의 IOU 임계값
WHEELCHAIR_DISTANCE_THRESHOLD = 50  # 휠체어와의 거리 임계값 (픽셀)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def calculate_size(box):
    return (box[2] - box[0]) * (box[3] - box[1])




def initialize_furniture(first_frame):
    global furniture_info
    furniture_results = furniture_model(first_frame)
    for r in furniture_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            if cls in safe_furniture_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                furniture_info.append({'box': (x1, y1, x2, y2), 'class': cls})
    print(f"가구 감지 완료: {len(furniture_info)}개의 안전한 가구 감지됨")


def assign_object_id(current_objects):
    global last_object_positions, object_id_counter

    new_object_positions = {}

    for obj in current_objects:
        best_match_id = None
        best_match_distance = float('inf')

        for last_id, last_pos in last_object_positions.items():
            distance = calculate_distance(obj['center'], last_pos)
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_id = last_id

        if best_match_id is not None and best_match_distance < 50:  # 50 픽셀 이내의 거리를 같은 객체로 간주
            obj['id'] = best_match_id
            new_object_positions[best_match_id] = obj['center']
        else:
            object_id_counter += 1
            obj['id'] = object_id_counter
            new_object_positions[object_id_counter] = obj['center']

    last_object_positions = new_object_positions
    return current_objects


def is_near_wheelchair(person_box, wheelchairs):
    person_center = ((person_box[0] + person_box[2]) // 2, (person_box[1] + person_box[3]) // 2)
    for wheelchair in wheelchairs:
        wheelchair_box = wheelchair['box']
        wheelchair_center = ((wheelchair_box[0] + wheelchair_box[2]) // 2, (wheelchair_box[1] + wheelchair_box[3]) // 2)

        iou = calculate_iou(person_box, wheelchair_box)
        distance = calculate_distance(person_center, wheelchair_center)

        if iou > WHEELCHAIR_IOU_THRESHOLD or distance < WHEELCHAIR_DISTANCE_THRESHOLD:
            return True
    return False


def process_frame(frame, frame_count):
    global previous_positions

    detected_objects = {'person': [], 'wheelchair': []}

    # 사람과 휠체어 감지
    person_wheelchair_results = person_wheelchair_model(frame)
    for r in person_wheelchair_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 0:  # person
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                detected_objects['person'].append({'box': (x1, y1, x2, y2), 'center': center})
            elif cls == 1:  # wheelchair
                detected_objects['wheelchair'].append({'box': (x1, y1, x2, y2)})

    # 객체 ID 할당
    detected_objects['person'] = assign_object_id(detected_objects['person'])

    # 안전 분석
    for person in detected_objects['person']:
        i = person['id']
        current_pos = person['center']
        current_size = calculate_size(person['box'])

        if i not in previous_positions:
            previous_positions[i] = {
                'positions': deque([current_pos], maxlen=POSITION_HISTORY_LENGTH),
                'sizes': deque([current_size], maxlen=POSITION_HISTORY_LENGTH),
                'stationary_count': 0,
                'unsafe_count': 0,
                'status': 'Moving'  # 초기 상태를 'Moving'으로 설정
            }
        else:
            previous_positions[i]['positions'].append(current_pos)
            previous_positions[i]['sizes'].append(current_size)

        # 평균 이동 거리 계산
        if len(previous_positions[i]['positions']) > 1:
            avg_distance = np.mean(
                [calculate_distance(current_pos, pos) for pos in list(previous_positions[i]['positions'])[:-1]])
        else:
            avg_distance = 0

        # 크기 변화 계산
        if len(previous_positions[i]['sizes']) > 1:
            size_change = abs(current_size - np.mean(list(previous_positions[i]['sizes'])[:-1])) / np.mean(
                list(previous_positions[i]['sizes'])[:-1])
        else:
            size_change = 0

        is_moving = avg_distance > MOVEMENT_THRESHOLD or size_change > SIZE_CHANGE_THRESHOLD

        # 안전 여부 확인
        is_safe = False
        safe_object = None
        for furniture in furniture_info:
            if calculate_iou(person['box'], furniture['box']) > 0.3:
                is_safe = True
                safe_object = class_names[furniture['class']]
                break

        if not is_safe and is_near_wheelchair(person['box'], detected_objects['wheelchair']):
            is_safe = True
            safe_object = 'wheelchair'

        # 상태 결정 (우선순위: Safe > Moving > Unsafe)
        if is_safe:
            new_status = f"Safe ({safe_object})"
            color = (0, 255, 0)  # 녹색
            previous_positions[i]['stationary_count'] = 0
            previous_positions[i]['unsafe_count'] = 0
        elif is_moving:
            new_status = "Moving"
            color = (0, 255, 255)  # 노란색
            previous_positions[i]['stationary_count'] = 0
            previous_positions[i]['unsafe_count'] = 0
        else:
            previous_positions[i]['stationary_count'] += 1
            if previous_positions[i]['stationary_count'] > STATIONARY_THRESHOLD:
                previous_positions[i]['unsafe_count'] += 1
                if previous_positions[i]['unsafe_count'] > UNSAFE_THRESHOLD:
                    new_status = "UNSAFE"
                    color = (0, 0, 255)  # 빨간색
                else:
                    new_status = "Caution (Not Moving)"
                    color = (0, 165, 255)  # 주황색
            else:
                new_status = "Stationary"
                color = (255, 165, 0)  # 연한 주황색

        previous_positions[i]['status'] = new_status

        x1, y1, x2, y2 = person['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # UNSAFE 상태일 때 특별한 태그 추가
        if new_status == "UNSAFE":
            cv2.putText(frame, "UNSAFE", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"ID {i}: {new_status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"ID {i}: Status: {new_status}, Avg Distance: {avg_distance:.2f}, Size Change: {size_change:.2f}, "
              f"Is Moving: {is_moving}, Stationary count: {previous_positions[i]['stationary_count']}, "
              f"Unsafe count: {previous_positions[i]['unsafe_count']}")


    # 가구 정보 표시
    for furniture in furniture_info:
        x1, y1, x2, y2 = furniture['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Safe: {class_names[furniture['class']]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    cv2.putText(frame, f"Frame: {frame_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


# 메인 실행 코드
input_video_path = '/tennis/original/test-video.mp4'
output_video_path = '/tennis/output/v3-output.mp4'

cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

ret, first_frame = cap.read()
if ret:
    initialize_furniture(first_frame)

frame_count = 0
skip_frames = 1  # 프레임 건너뛰기 설정

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    processed_frame = process_frame(frame, frame_count)
    out.write(processed_frame)

    cv2.imshow('Processing Video', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("비디오 처리 완료")