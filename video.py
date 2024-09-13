import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

# GPU 사용 가능 여부 확인 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# YOLO v8 모델 로드
model = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/4-1/best.pt')

# 객체 추적을 위한 딕셔너리
tracked_objects = {}

# 클래스별 박스 정보를 저장할 딕셔너리
class_boxes = {
    0: [],  # person
    1: [],  # wheelchair
    2: []  # notwheelchair
}


def calculate_distance(box1, box2):
    # 박스의 중심점 계산
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)

    # 유클리드 거리 계산
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def analyze_interactions(persons, wheelchairs, threshold=200):
    interactions = []
    for p_idx, person in enumerate(persons):
        for w_idx, wheelchair in enumerate(wheelchairs):
            distance = calculate_distance(person['box'], wheelchair['box'])
            if distance < threshold:
                interactions.append({
                    'person_idx': p_idx,
                    'wheelchair_idx': w_idx,
                    'distance': distance
                })
    return interactions


def process_frame(frame, frame_count):
    # 프레임별 클래스 박스 정보 초기화
    for cls in class_boxes:
        class_boxes[cls] = []

    results = model(frame)

    current_time = time.time()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf)
            cls = int(box.cls)

            # 클래스에 따른 레이블 설정
            if cls == 0:
                label = "Person"
            elif cls == 1:
                label = "Wheelchair"
            elif cls == 2:
                label = "NotWheelchair"
            else:
                label = f"Class {cls}"

            # 클래스별 박스 정보 저장
            class_boxes[cls].append({
                'box': (x1, y1, x2, y2),
                'conf': conf
            })

            color = (0, 255, 0)  # 기본 색상 (녹색)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 상호작용 분석
    interactions = analyze_interactions(class_boxes[0], class_boxes[1])

    # 상호작용 시각화 및 정보 출력
    for interaction in interactions:
        person = class_boxes[0][interaction['person_idx']]
        wheelchair = class_boxes[1][interaction['wheelchair_idx']]

        # 상호작용 선 그리기
        p_center = ((person['box'][0] + person['box'][2]) // 2, (person['box'][1] + person['box'][3]) // 2)
        w_center = (
        (wheelchair['box'][0] + wheelchair['box'][2]) // 2, (wheelchair['box'][1] + wheelchair['box'][3]) // 2)
        cv2.line(frame, p_center, w_center, (255, 0, 0), 2)

        # 상호작용 정보 출력
        print(f"Frame {frame_count} - Interaction: Person-Wheelchair, Distance: {interaction['distance']:.2f}")

    cv2.putText(frame, f"Frame: {frame_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame


# 입력 및 출력 비디오 파일 경로
input_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/original/tennis.mp4'
output_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/output/tennis_output.mp4'

cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
skip_frames = 4  # 프레임 건너뛰기 설정

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