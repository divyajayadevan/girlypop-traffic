import cv2
from ultralytics import YOLO

class TrafficProcessor:
    def __init__(self, model_path='yolov8n.pt', confidence=0.35):
        self.model = YOLO(model_path)
        self.confidence = confidence
        # COCO class IDs: 2=Car, 3=Bike, 5=Bus, 7=Truck
        self.class_names = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
        
    def process_frame(self, frame, current_counts, counted_ids):
        # 1. OPTIMIZATION: Resize frame to improve inference speed
        # YOLOv8 is trained on 640x640. Feeding 1080p slows it down without much gain.
        original_h, original_w = frame.shape[:2]
        new_w = 640
        scale = new_w / original_w
        new_h = int(original_h * scale)
        
        small_frame = cv2.resize(frame, (new_w, new_h))
        
        # 2. Run Tracking on the small frame
        results = self.model.track(small_frame, persist=True, conf=self.confidence, verbose=False)
        
        # 3. Process Detections
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                # --- VISUALIZATION ---
                # Draw Box (Scale coordinates back to original frame size)
                x1, y1, x2, y2 = (box / scale).astype(int)
                
                # Color coding (Green for tracked, Red for unknown)
                color = (0, 255, 0) 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                raw_label = self.model.names[cls_id]
                label_text = f"{raw_label} #{track_id}"
                cv2.putText(frame, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- COUNTING LOGIC (Instant & Unique) ---
                if track_id not in counted_ids:
                    # Filter for our target classes only
                    if raw_label in ['car', 'motorcycle', 'bus', 'truck']:
                        clean_label = 'Bike' if raw_label == 'motorcycle' else raw_label.capitalize()
                        
                        current_counts[clean_label] += 1
                        counted_ids.add(track_id) # Mark as counted forever

        return frame, current_counts, counted_ids