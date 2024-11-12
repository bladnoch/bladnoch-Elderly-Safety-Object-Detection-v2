import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

# GPU 사용 가능 여부 확인 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# furniture_model 이 시작부터 끝까지 같이 돌아가는 문제가 있음

# 두 개의 YOLO 모델 로드
furniture_model = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/original/yolov8m.pt')  # 기본 pre-trained 모델
person_wheelchair_model = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/original/yolov8m.pt')  # 기본 pre-trained 모델

# 안전한 가구 클래스 ID (COCO 데이터셋 기준)
safe_furniture_classes = [56, 57, 59]  # 56: chair, 57: couch, 59: bed

# 클래스 이름 매핑
class_names = {
    56: 'chair',
    57: 'couch',
    59: 'bed'
}

# 객체 정보를 저장할 딕셔너리
detected_objects = {
    'furniture': [],
    'person': [],
    'wheelchair': []
}


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


def process_frame(frame, frame_count):
    # 객체 정보 초기화
    detected_objects['furniture'] = []
    detected_objects['person'] = []
    detected_objects['wheelchair'] = []

    # 1단계: 가구 감지
    furniture_results = furniture_model(frame)
    for r in furniture_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            if cls in safe_furniture_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_objects['furniture'].append({
                    'box': (x1, y1, x2, y2),
                    'class': cls
                })
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Safe: {class_names[cls]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

    # 2단계: 사람과 휠체어 감지
    person_wheelchair_results = person_wheelchair_model(frame)
    for r in person_wheelchair_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 0:  # person
                detected_objects['person'].append({
                    'box': (x1, y1, x2, y2)
                })
            elif cls == 1:  # wheelchair
                detected_objects['wheelchair'].append({
                    'box': (x1, y1, x2, y2)
                })

    # 안전 분석
    for person in detected_objects['person']:
        is_safe = False
        safe_object = None
        for furniture in detected_objects['furniture']:
            if calculate_iou(person['box'], furniture['box']) > 0.3:  # 30% 이상 겹치면 안전하다고 판단
                is_safe = True
                safe_object = class_names[furniture['class']]
                break
        for wheelchair in detected_objects['wheelchair']:
            if calculate_iou(person['box'], wheelchair['box']) > 0.3:
                is_safe = True
                safe_object = 'wheelchair'
                break

        x1, y1, x2, y2 = person['box']
        color = (0, 255, 0) if is_safe else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        status = f"Safe ({safe_object})" if is_safe else "Unsafe"
        cv2.putText(frame, f"Person: {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Frame: {frame_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


# 입력 및 출력 비디오 파일 경로
input_video_path = '/Users/doungukkim/Desktop/workspace/object-detecting-v2/tennis/original/one-not-moving.mp4'
output_video_path = '/tennis/output/etc/one-not-moving.mp4'

cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
skip_frames = 1  # 프레임 건너뛰기 설정

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