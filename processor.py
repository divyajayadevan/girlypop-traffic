import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class TrafficProcessor:
    def __init__(self, model_path='yolov8n.pt', confidence=0.35):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_names = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
        
        # Store previous positions to calculate direction: {track_id: (y_prev, y_curr)}
        self.track_history = defaultdict(lambda: []) 

    def process_frame(self, frame, current_counts, counted_ids):
        # 1. 1080p Processing: No resize. We work on the full frame.
        height, width = frame.shape[:2]
        
        # Define the "Gate Line" at 60% height
        line_y = int(height * 0.6)
        
        # Run Tracking
        results = self.model.track(frame, persist=True, conf=self.confidence, verbose=False)
        
        # Visuals: Draw the Gate Line
        # Cyan color for the line
        cv2.line(frame, (0, line_y), (width, line_y), (255, 255, 0), 2)
        cv2.putText(frame, "DETECTION GATE", (10, line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                # Calculate Centroid
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                
                # --- DIRECTIONAL LOGIC ---
                # We need history to know if it's moving UP or DOWN
                prev_y = cy # Default to current if no history
                
                # Get history for this ID
                history = self.track_history[track_id]
                if len(history) > 0:
                    prev_y = history[-1]
                
                # Update history (keep only last 5 frames to save memory)
                history.append(cy)
                if len(history) > 5: history.pop(0)
                
                # --- CROSSING LOGIC ---
                # Check if it crossed the line in this exact frame step
                
                # Case 1: Moving DOWN (Incoming)
                # Previous y was above line (< line_y), Current y is below line (>= line_y)
                if prev_y < line_y and cy >= line_y:
                    if track_id not in counted_ids:
                        self._update_counts(cls_id, current_counts, "Incoming")
                        counted_ids.add(track_id)
                        # Visual Flash Green
                        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 5)

                # Case 2: Moving UP (Outgoing)
                # Previous y was below line (> line_y), Current y is above line (<= line_y)
                elif prev_y > line_y and cy <= line_y:
                    if track_id not in counted_ids:
                        self._update_counts(cls_id, current_counts, "Outgoing")
                        counted_ids.add(track_id)
                        # Visual Flash Red (or Magenta)
                        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 255), 5)

                # --- VISUALIZATION ---
                raw_label = self.model.names[cls_id]
                color = (0, 255, 0) if track_id in counted_ids else (0, 0, 255)
                
                # Draw Box & Label
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, f"{raw_label} #{track_id}", (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw centroid
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        return frame, current_counts, counted_ids

    def _update_counts(self, cls_id, counts, direction):
        raw_label = self.model.names[cls_id]
        if raw_label in ['car', 'motorcycle', 'bus', 'truck']:
            cat = 'Bike' if raw_label == 'motorcycle' else raw_label.capitalize()
            # Key format: "Incoming_Car" or "Outgoing_Bus"
            key = f"{direction}_{cat}"
            counts[key] += 1