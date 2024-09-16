import cv2
import torch
from ultralytics import YOLO
import numpy as np

# GPU 사용 가능 여부 확인 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 두 개의 YOLO 모델 로드
furniture_model = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/original/yolov8m.pt')
person_wheelchair_model = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/4-1/best.pt')

# 안전한 가구 클래스 ID (COCO 데이터셋 기준)
safe_furniture_classes = [56, 57, 59]  # 56: chair, 57: couch, 59: bed

# 클래스 이름 매핑
class_names = {
    56: 'chair',
    57: 'couch',
    59: 'bed'
}

# 전역 변수로 가구 정보 저장
furniture_info = []

# 사람의 이전 위치를 저장할 딕셔너리
previous_positions = {}

# 움직임 감지를 위한 임계값 (픽셀 단위)
MOVEMENT_THRESHOLD = 10  # 더 작은 움직임도 감지

# 움직임이 없는 상태로 간주할 시간 (프레임 수)
STATIONARY_THRESHOLD = 30 * 1  # 2초 (30fps 기준)

# Unsafe 상태로 간주할 시간 (프레임 수)
UNSAFE_THRESHOLD = 30 * 3  # 5초 (30fps 기준)

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

    return intersection / union if union > 0 else 0

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
                    'class': cls
                })
    print(f"가구 감지 완료: {len(furniture_info)}개의 안전한 가구 감지됨")

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
                detected_objects['person'].append({
                    'box': (x1, y1, x2, y2),
                    'center': center
                })
            elif cls == 1:  # wheelchair
                detected_objects['wheelchair'].append({
                    'box': (x1, y1, x2, y2)
                })

    # 안전 분석
    for i, person in enumerate(detected_objects['person']):
        is_safe = False
        safe_object = None
        is_moving = True

        # 이전 위치와 비교하여 움직임 확인
        if i in previous_positions:
            prev_pos, stationary_count = previous_positions[i]['position'], previous_positions[i]['stationary_count']
            current_pos = person['center']
            distance = calculate_distance(prev_pos, current_pos)

            print(f"Person {i}: Distance moved: {distance:.2f}, Stationary count: {stationary_count}")

            if distance < MOVEMENT_THRESHOLD:
                stationary_count += 1
                if stationary_count > STATIONARY_THRESHOLD:
                    is_moving = False
            else:
                stationary_count = 0

            previous_positions[i]['position'] = current_pos
            previous_positions[i]['stationary_count'] = stationary_count
        else:
            previous_positions[i] = {'position': person['center'], 'stationary_count': 0, 'unsafe_count': 0}
            print(f"Person {i}: First detection")

        for furniture in furniture_info:
            if calculate_iou(person['box'], furniture['box']) > 0.5:  # IoU 임계값을 높임
                is_safe = True
                safe_object = class_names[furniture['class']]
                break
        for wheelchair in detected_objects['wheelchair']:
            if calculate_iou(person['box'], wheelchair['box']) > 0.5:  # IoU 임계값을 높임
                is_safe = True
                safe_object = 'wheelchair'
                break

        x1, y1, x2, y2 = person['box']
        if is_safe:
            color = (0, 255, 0)  # 녹색 (안전)
            status = f"Safe ({safe_object})"
            previous_positions[i]['unsafe_count'] = 0
        elif is_moving:
            color = (0, 255, 255)  # 노란색 (움직이는 중)
            status = "Moving"
            previous_positions[i]['unsafe_count'] = 0
        else:
            previous_positions[i]['unsafe_count'] += 1
            if previous_positions[i]['unsafe_count'] > UNSAFE_THRESHOLD:
                color = (0, 0, 255)  # 빨간색 (위험)
                status = "Unsafe (Not Moving for long time)"
            else:
                color = (0, 165, 255)  # 주황색 (주의)
                status = "Caution (Not Moving)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Person {i}: {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"Person {i}: Status: {status}, Is moving: {is_moving}, Is safe: {is_safe}, Unsafe count: {previous_positions[i]['unsafe_count']}")

    # 가구 정보 표시 (항상 표시)
    for furniture in furniture_info:
        x1, y1, x2, y2 = furniture['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Safe: {class_names[furniture['class']]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    cv2.putText(frame, f"Frame: {frame_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# 입력 및 출력 비디오 파일 경로
input_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/original/test-video.mp4'
output_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/output/test-video.mp4'

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
skip_frames = 5  # 프레임 건너뛰기 설정

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 일정 수의 프레임을 건너뛰기
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