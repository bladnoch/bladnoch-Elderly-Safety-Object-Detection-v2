import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

# YOLO v8 모델 로드
model = YOLO('/Users/doungukkim/Desktop/workspace/object-detecting-v2/trained-models/4-1/best.pt')

# 객체 추적을 위한 딕셔너리
tracked_objects = {}

# 프레임 버퍼 (최근 5개의 처리된 프레임을 저장)
frame_buffer = deque(maxlen=5)


def process_frame(frame, frame_count):
    results = model(frame)

    current_time = time.time()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf)
            cls = int(box.cls)

            if cls == 0:  # 'person' 클래스
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                obj_id = f"person_{frame_count}_{center}"

                color = (0, 255, 0)  # 기본 색상 (녹색)

                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = {
                        'last_movement_time': current_time,
                        'last_position': center
                    }
                else:
                    last_position = tracked_objects[obj_id]['last_position']
                    if calculate_distance(last_position, center) > 10:  # 움직임 감지 임계값
                        tracked_objects[obj_id]['last_movement_time'] = current_time
                        tracked_objects[obj_id]['last_position'] = center
                        color = (0, 255, 0)  # 녹색 (움직임 감지)
                    else:
                        time_since_last_movement = current_time - tracked_objects[obj_id]['last_movement_time']
                        if time_since_last_movement > 10:  # 10초 이상 움직임이 없으면 경고
                            color = (0, 0, 255)  # 빨간색 (경고)
                            cv2.putText(frame, f"Alert: No movement for {time_since_last_movement:.1f}s",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            log_alert(obj_id, time_since_last_movement)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Frame: {frame_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def calculate_distance(pos1, pos2):
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def log_alert(obj_id, duration):
    print(f"경고: 객체 ID {obj_id}가 {duration:.2f}초 동안 움직이지 않았습니다.")


# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # 카메라 FPS 설정 (지원되는 경우)

frame_count = 0
display_fps = 30  # 화면 출력 FPS
frame_time = 1 / display_fps

last_display_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 모든 프레임에 대해 처리 수행
    processed_frame = process_frame(frame, frame_count)
    frame_buffer.append(processed_frame)

    current_time = time.time()
    if current_time - last_display_time >= frame_time:
        # 화면 출력
        if frame_buffer:
            display_frame = frame_buffer[-1]  # 가장 최근에 처리된 프레임 사용
            cv2.imshow('Elderly Monitoring', display_frame)
        last_display_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("모니터링 종료")