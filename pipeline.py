import io
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import ConvNextImageProcessor, ConvNextForImageClassification

class AnimalDetectionPipeline:
    def __init__(self):
        # ==========================================
        # TODO: REPLACE WITH YOUR OWN MODEL PATHS
        # ==========================================
        self.megadetector_path = "yolov8n.pt" # Example path, replace with real weights
        self.convnext_path = "microsoft/convnext-tiny-224" # Example path, replace with real weights
        
        print("Loading models (this might take a moment)...")
        # Load Stage 1: MegaDetector (YOLOv8 based for this demo)
        self.detector = YOLO(self.megadetector_path)
        
        # Load Stage 2: ConvNeXt Classifier
        self.processor = ConvNextImageProcessor.from_pretrained(self.convnext_path)
        self.classifier = ConvNextForImageClassification.from_pretrained(self.convnext_path)
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.to(self.device)
        self.classifier.eval()
        print(f"Models loaded on {self.device}")

    def predict(self, image_bytes: bytes):
        """
        Runs the 2-stage pipeline on the input image using bytes.
        Returns bounding boxes and top-3 classification results.
        """
        # Load Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        
        # Stage 1: Detection
        # Run inference
        det_results = self.detector(image, verbose=False)[0]
        
        boxes = []
        best_crop = None
        max_area = 0
        best_box_coords = None
        
        # Process detection results (find the primary animal)
        if len(det_results.boxes) > 0:
            for box in det_results.boxes:
                # box.xyxy format is [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                # Filter out low confidence or non-animal classes if needed
                
                boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf,
                    "class_id": cls_id
                })
                
                # Heuristic: classify the largest detected object
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box_coords = (x1, y1, x2, y2)
                    
            if best_box_coords:
                # Crop the image for stage 2
                best_crop = image.crop(best_box_coords)
        
        # If no box found, fallback to classifying the whole image
        if best_crop is None:
            best_crop = image
            
        # Stage 2: Classification (ConvNeXt)
        inputs = self.processor(best_crop, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            logits = outputs.logits
            
        # Get Top-3 Predictions
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        top3_indices = torch.topk(probs, 3).indices.tolist()
        top3_probs = torch.topk(probs, 3).values.tolist()
        
        top_predictions = []
        for idx, prob in zip(top3_indices, top3_probs):
            label = self.classifier.config.id2label.get(idx, f"Class {idx}")
            # Format label nicely (e.g., removing technical prefixes if any)
            formatted_label = str(label).split(',')[0].title()
            top_predictions.append({
                "label": formatted_label,
                "confidence": round(prob * 100, 2)
            })
            
        return {
            "boxes": boxes, # All detected boxes
            "top_predictions": top_predictions # Classification for the primary object
        }
