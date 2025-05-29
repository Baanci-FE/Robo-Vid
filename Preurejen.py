#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import mediapipe as mp
import time
from collections import deque
import os
from GITHUB.marking import CornerMarker


def process_video(video_path: str) -> Dict[str, List[str]]:
    X = 5
    threshold_speed = 70
    playback_speed = 1
    frame_delay = int((1.0 / playback_speed) * 33)
    effective_threshold = threshold_speed * playback_speed

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        exit()

    prev_positions = {5: None, 9: None, 13: None, 17: None}
    velocity_history = {5: deque(maxlen=X), 9: deque(maxlen=X), 13: deque(maxlen=X), 17: deque(maxlen=X)}

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        exit()

    marker = CornerMarker(first_frame)
    cv2.namedWindow("Click 4 Corners")
    cv2.setMouseCallback("Click 4 Corners", marker.click_event)

    while True:
        cv2.imshow("Click 4 Corners", marker.img_copy)
        key = cv2.waitKey(1) & 0xFF
        if marker.result_ready and (key == 13 or key == 32):
            break
        if key == 27:
            cv2.destroyAllWindows()
            cap.release()
            exit()

    points, (mid1, mid2) = marker.get_points_and_midline()
    cv2.destroyWindow("Click 4 Corners")

    grab_square = np.array([points[0], points[1], mid2, mid1])
    drop_square = np.array([points[3], points[2], mid2, mid1])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    pTime = time.time()
    current_gesture = "prazna_roka"
    new_gesture = "prazna_roka"
    frame_count = 0
    roka_polna = 0
    space_pressed = False
    output_dict = {}

    def is_point_inside_polygon(point, polygon):
        if point is None or polygon is None:
            return False
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    while True:
        frame_count += 1
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        current_positions = {}
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id in [5, 9, 13, 17]:
                        current_positions[id] = (cx, cy)

                for id in [5, 9, 13, 17]:
                    if id in current_positions and prev_positions[id] is not None:
                        prev_x, prev_y = prev_positions[id]
                        curr_x, curr_y = current_positions[id]
                        displacement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                        time_diff = (time.time() - pTime) / playback_speed if pTime != 0 else 1 / 30
                        velocity = displacement / time_diff
                        velocity_history[id].append(velocity)

                for id in current_positions:
                    prev_positions[id] = current_positions[id]

        moving_averages = {
            id: sum(velocity_history[id]) / len(velocity_history[id]) if velocity_history[id] else 0
            for id in [5, 9, 13, 17]
        }

        cTime = time.time()
        pTime = cTime

        pos_9 = current_positions.get(9)
        pos_13 = current_positions.get(13)
        in_grab_area = pos_9 and pos_13 and (
            is_point_inside_polygon(pos_13, grab_square) and is_point_inside_polygon(pos_9, grab_square))
        in_drop_area = pos_9 and pos_13 and (
            is_point_inside_polygon(pos_13, drop_square) and is_point_inside_polygon(pos_9, drop_square))

        avg_velocity = sum(moving_averages.values()) / 4 if moving_averages else 0

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == 32:
            space_pressed = True

        if not space_pressed:
            if avg_velocity > effective_threshold:
                new_gesture = "prenos_pina" if roka_polna == 1 else "prazna_roka"
            elif in_grab_area:
                new_gesture = "prijemanje_pina"
                roka_polna = 1
            elif in_drop_area:
                new_gesture = "odlaganje_pina"
                roka_polna = 0
        else:
            if in_drop_area:
                roka_polna = 0
            elif in_grab_area:
                roka_polna = 1
            new_gesture = "prenos_pina" if roka_polna == 1 else "prazna_roka"

        output_dict[str(frame_count)] = [new_gesture]
        current_gesture = new_gesture

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return output_dict


def load_json(file_path: str):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        sys.exit(1)


def convert_to_frame_events(annotations: List[Dict]) -> Dict[int, List[str]]:
    frame_events = {}
    for annotation in annotations:
        frame_start = annotation["start_frame"]
        frame_stop = annotation["end_frame"]
        label = annotation["label"].strip()
        for frame in range(frame_start, frame_stop + 1):
            frame_events[frame] = [label]
    return frame_events


def create_confusion_matrix(ground_truth: Dict[int, List[str]], student_output: Dict[str, List[str]],
                            events: List[str]) -> np.ndarray:
    student_output_int = {int(k): v for k, v in student_output.items()}
    n_events = len(events)
    conf_matrix = np.zeros((n_events, n_events), dtype=int)
    event_to_idx = {event: idx for idx, event in enumerate(events)}

    for frame_id, gt_events in ground_truth.items():
        if frame_id not in student_output_int:
            continue
        gt_event = gt_events[0]
        pred_event = student_output_int[frame_id][0]
        gt_idx = event_to_idx[gt_event]
        pred_idx = event_to_idx[pred_event]
        conf_matrix[gt_idx][pred_idx] += 1

    return conf_matrix


def plot_confusion_matrix(conf_matrix: np.ndarray, events: List[str], output_path: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=events, yticklabels=events)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Event')
    plt.ylabel('True Event')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_metrics(ground_truth: Dict[int, List[str]], student_output: Dict[str, List[str]]) -> Dict:
    student_output_int = {int(k): v for k, v in student_output.items()}
    metrics = {
        "total_frames": len(ground_truth),
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "missing_predictions": 0,
        "extra_predictions": 0
    }

    for frame_id, gt_events in ground_truth.items():
        if frame_id not in student_output_int:
            metrics["missing_predictions"] += 1
            continue
        student_events = student_output_int[frame_id]
        if set(gt_events) == set(student_events):
            metrics["correct_predictions"] += 1
        else:
            metrics["incorrect_predictions"] += 1

    metrics["extra_predictions"] = len(student_output_int) - len(ground_truth)
    total_predictions = metrics["correct_predictions"] + metrics["incorrect_predictions"]
    metrics["accuracy"] = metrics["correct_predictions"] / total_predictions if total_predictions > 0 else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Validate student video processing algorithm')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('ground_truth_path', type=str, help='Path to the ground truth JSON file')
    parser.add_argument('student_output_path', type=str, help='Path to the student output JSON file')
    args = parser.parse_args()

    # Load ground truth JSON
    ground_truth_data = load_json(args.ground_truth_path)
    print("Ground Truth Data:", ground_truth_data)  # This line is just for debugging, optional

    # ✅ FIXED CONDITION ORDER AND INDENTATION
    if isinstance(ground_truth_data, list):
        annotation_list = ground_truth_data
    elif isinstance(ground_truth_data, dict) and "annotations" in ground_truth_data:
        annotation_list = ground_truth_data["annotations"]
    else:
        print("Error: ground_truth_data is not in the expected format.")
        sys.exit(1)

    # Convert to per-frame format
    ground_truth = convert_to_frame_events(annotation_list)

    # Process video
    student_output = process_video(args.video_path)

    # Save predictions
    with open(args.student_output_path, 'w') as f:
        json.dump(student_output, f, indent=4)

    # Evaluate
    all_events = sorted(set(event for events in ground_truth.values() for event in events))
    conf_matrix = create_confusion_matrix(ground_truth, student_output, all_events)
    output_path = Path(args.student_output_path).stem + "_confusion_matrix.png"
    plot_confusion_matrix(conf_matrix, all_events, output_path)
    print(f"\nConfusion matrix saved to: {output_path}")

    metrics = calculate_metrics(ground_truth, student_output)
    print("\nValidation Results:")
    print("-" * 50)
    print(f"Total frames processed: {metrics['total_frames']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Incorrect predictions: {metrics['incorrect_predictions']}")
    print(f"Missing predictions: {metrics['missing_predictions']}")
    print(f"Extra predictions: {metrics['extra_predictions']}")
    print(f"Overall accuracy: {metrics['accuracy']:.2%}")
    print("-" * 50)

if __name__ == "__main__":
    main()
