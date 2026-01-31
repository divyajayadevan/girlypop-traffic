import cv2
from ultralytics import YOLO

class TrafficProcessor:
    def __init__(self, model_path='yolov8n.pt', confidence=0.45):
        self.model = YOLO(model_path)
        self.confidence = confidence
        # COCO class IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.class_names = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
        
    def process_frame(self, frame, line_position, current_counts):
        height, width = frame.shape[:2]
        line_y = int(height * line_position)
        
        # Run tracking with persistence to keep IDs consistent
        results = self.model.track(frame, persist=True, conf=self.confidence, verbose=False)
        
        # Draw the counting line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                # Calculate centroid of the object
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                
                # Visual feedback for the centroid
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # Detection Logic: Check if centroid is within a 20px buffer of the line
                if line_y - 10 < cy < line_y + 10:
                    raw_label = self.model.names[cls_id]
                    
                    if raw_label in ['car', 'motorcycle', 'bus', 'truck']:
                        label = 'Bike' if raw_label == 'motorcycle' else raw_label.capitalize()
                        current_counts[label] += 1

        return frame, current_counts