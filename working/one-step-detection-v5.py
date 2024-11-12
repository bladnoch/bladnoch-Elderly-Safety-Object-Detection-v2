import cv2
import torch
from ultralytics import YOLO
import numpy as np
from sort import Sort  # SORT 알고리즘을 위한 모듈
from collections import deque

# GPU 사용 가능 여부 확인 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 두 개의 YOLO 모델 로드 (경로를 실제 모델 파일 위치로 수정하세요)
furniture_model = YOLO('/path/to/your/furniture_model.pt')
person_wheelchair_model = YOLO('/path/to/your/person_wheelchair_model.pt')

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
        self.positions = deque(maxlen=30)  # 최근 위치들 저장
        self.positions.append(center)
        self.stationary_count = 0
        self.moving_count = 0
        self.state = "Initializing"
        self.near_safe_object = False

# 전역 변수들
tracked_objects = {}
tracker = Sort()  # SORT 객체 추적기 초기화

# 프레임률을 가져오기 위해 비디오 캡처 초기화 (경로를 실제 비디오 파일 위치로 수정하세요)
input_video_path = '/path/to/your/input_video.mp4'
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.release()

# 상수 정의 (fps 기반으로 설정)
MOVEMENT_THRESHOLD = 2  # 픽셀 단위
STATIONARY_THRESHOLD = fps * 2  # 2초 동안 정지하면 'Stationary'로 판단
MOVING_THRESHOLD = fps // 2     # 0.5초 동안 움직이면 'Moving'으로 판단
SAFE_DISTANCE = 100  # 안전 객체와의 최소 거리 (픽셀 단위)

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

def calculate_min_distance(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    left = x2_max < x1_min
    right = x1_max < x2_min
    bottom = y2_max < y1_min
    top = y1_max < y2_min

    if top and left:
        return calculate_distance((x1_min, y1_max), (x2_max, y2_min))
    elif left and bottom:
        return calculate_distance((x1_min, y1_min), (x2_max, y2_max))
    elif bottom and right:
        return calculate_distance((x1_max, y1_min), (x2_min, y2_max))
    elif right and top:
        return calculate_distance((x1_max, y1_max), (x2_min, y2_min))
    elif left:
        return x1_min - x2_max
    elif right:
        return x2_min - x1_max
    elif bottom:
        return y1_min - y2_max
    elif top:
        return y2_min - y1_max
    else:
        return 0  # 박스가 겹치는 경우

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
        return obj.state  # 위치 정보가 충분하지 않으면 상태 유지

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
        distance = calculate_min_distance(obj.box, furniture['box'])
        if distance < SAFE_DISTANCE:
            return True
    return False

def process_frame(frame, frame_count):
    global tracker, tracked_objects

    # 사람 감지
    person_results = person_wheelchair_model(frame)

    detections = []
    for r in person_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            if cls == 0:  # person 클래스만 추적
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf)
                detections.append([x1, y1, x2, y2, score])

    # numpy 배열로 변환
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    # 객체 추적
    tracks = tracker.update(detections)

    # 현재 프레임의 객체들
    current_tracked_objects = {}

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # 트랙 ID를 사용하여 객체 상태 관리
        if track_id in tracked_objects:
            tracked_obj = tracked_objects[track_id]
            tracked_obj.box = (x1, y1, x2, y2)
            tracked_obj.center = center
            tracked_obj.positions.append(center)
        else:
            tracked_obj = TrackedObject((x1, y1, x2, y2), center)

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
            else:
                color = (0, 0, 255)  # 빨간색
                status = f"Unsafe ({tracked_obj.state})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Person {int(track_id)}: {status}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 이전 프레임의 객체 정보를 현재 프레임의 객체로 업데이트
    tracked_objects = current_tracked_objects

    # 가구 정보 표시
    for furniture in furniture_info:
        x1, y1, x2, y2 = furniture['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Safe: {class_names[furniture['class']]}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Frame: {frame_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# 메인 실행 부분
def main():
    global tracker, tracked_objects

    # 입력 및 출력 비디오 경로를 실제 파일 위치로 수정하세요
    input_video_path = '/path/to/your/input_video.mp4'
    output_video_path = '/path/to/your/output_video.mp4'

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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

        # 화면에 결과를 표시하려면 다음 주석을 해제하세요
        # cv2.imshow('Processing Video', processed_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("비디오 처리 완료")

if __name__ == "__main__":
    main()
