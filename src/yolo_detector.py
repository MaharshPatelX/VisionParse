import torch
import numpy as np
from PIL import Image
from typing import List
import cv2

def get_yolo_model(model_path):
    """Load YOLO model from path"""
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model

def predict_yolo(model, image, box_threshold=0.25, imgsz=None, scale_img=False, iou_threshold=0.7):
    """Run YOLO prediction on image"""
    if scale_img and imgsz:
        result = model.predict(
            source=image,
            conf=box_threshold,
            imgsz=imgsz,
            iou=iou_threshold,
        )
    else:
        result = model.predict(
            source=image,
            conf=box_threshold,
            iou=iou_threshold,
        )
    
    boxes = result[0].boxes.xyxy  # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]
    
    return boxes, conf, phrases

def annotate_image(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], 
                  text_scale: float = 0.4, text_padding: int = 3, text_thickness: int = 1, thickness: int = 1) -> np.ndarray:
    """Annotate image with bounding boxes and simple ID labels using OpenCV"""
    annotated_frame = image_source.copy()
    
    if boxes.numel() > 0:  # Check if boxes is not empty
        # Convert boxes to numpy and ensure they're integers
        xyxy = boxes.cpu().numpy().astype(int)
        
        # Color palette for different boxes - using brighter colors for thin lines
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (255, 165, 0), (128, 0, 128), (0, 128, 255), (255, 128, 0)
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            # Choose color
            color = colors[i % len(colors)]
            
            # Draw thin bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Simple ID label (just the number)
            text = str(i)
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )
            
            # Draw small label background
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - text_padding * 2),
                (x1 + text_width + text_padding * 2, y1),
                color,
                -1
            )
            
            # Draw ID text
            cv2.putText(
                annotated_frame,
                text,
                (x1 + text_padding, y1 - text_padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (0, 0, 0),  # Black text for better visibility
                text_thickness
            )
    
    return annotated_frame

def detect_ui_elements(image_path, model_path="weights/icon_detect/model.pt", 
                      confidence_threshold=0.05, iou_threshold=0.1, 
                      text_scale=0.4):
    """
    Detect UI elements using YOLO model - simplified version
    
    Args:
        image_path: Path to screenshot or PIL Image
        model_path: Path to YOLO model
        confidence_threshold: Minimum confidence for detection
        iou_threshold: IoU threshold for NMS
        text_scale: Scale for annotation text
    
    Returns:
        List of detected boxes with coordinates and confidence
    """
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    
    # Load YOLO model
    model = get_yolo_model(model_path)
    
    # Run YOLO prediction
    boxes, conf, phrases = predict_yolo(
        model=model, 
        image=image, 
        box_threshold=confidence_threshold,
        iou_threshold=iou_threshold
    )
    
    # Convert to list format for easier handling
    detected_boxes = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        
        detected_boxes.append({
            'id': i + 1,
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf[i]),
            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            'type': 'icon',
            'interactivity': True,
            'content': None  # Will be filled by VLM analysis
        })
    
    return detected_boxes

def predict_and_save(model_path: str, image_path: str, output_path: str, 
                    box_threshold: float = 0.25, text_scale: float = 0.4):
    """
    Complete pipeline: load model, predict, annotate and save image
    
    Args:
        model_path: Path to YOLO model file (.pt)
        image_path: Path to input image
        output_path: Path to save annotated image
        box_threshold: Confidence threshold for detections
        text_scale: Scale for text labels
    """
    
    # Load YOLO model
    print(f"Loading YOLO model from: {model_path}")
    model = get_yolo_model(model_path)
    
    # Load image
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Run prediction
    print("Running YOLO prediction...")
    boxes, confidences, phrases = predict_yolo(
        model=model, 
        image=image, 
        box_threshold=box_threshold
    )
    
    print(f"Found {len(boxes)} detections")
    
    # Annotate image with thin lines and simple IDs
    print("Annotating image...")
    annotated_image = annotate_image(
        image_source=image_np,
        boxes=boxes,
        logits=confidences,
        phrases=phrases,
        text_scale=text_scale,
        text_padding=3,
        text_thickness=1,
        thickness=1  # Thin bounding box lines
    )
    
    # Save result
    print(f"Saving annotated image to: {output_path}")
    result_image = Image.fromarray(annotated_image)
    result_image.save(output_path)
    
    print("Done!")
    
    return annotated_image, boxes, confidences

def draw_boxes_on_image(image_path, boxes, output_path=None):
    """Draw bounding boxes on image for visualization using the new annotate_image function"""
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
    else:
        image_np = np.array(image_path)
    
    # Convert boxes to torch tensor format
    box_tensors = []
    confidences = []
    
    for box in boxes:
        x1, y1, x2, y2 = box['bbox']
        box_tensors.append([x1, y1, x2, y2])
        confidences.append(box['confidence'])
    
    if box_tensors:
        boxes_tensor = torch.tensor(box_tensors, dtype=torch.float32)
        conf_tensor = torch.tensor(confidences, dtype=torch.float32)
        phrases = [str(i) for i in range(len(boxes))]
        
        # Use the annotate_image function
        annotated_image = annotate_image(
            image_source=image_np,
            boxes=boxes_tensor,
            logits=conf_tensor,
            phrases=phrases,
            text_scale=0.4,
            text_padding=3,
            text_thickness=1,
            thickness=1
        )
        
        if output_path:
            result_image = Image.fromarray(annotated_image)
            result_image.save(output_path)
        
        return annotated_image
    else:
        return image_np

def crop_image_regions(image_path, boxes):
    """Crop image regions for VLM analysis"""
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        # Convert PIL to OpenCV format
        image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
    
    cropped_regions = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box['bbox']
        
        # Add padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        # Crop region
        cropped = image[y1:y2, x1:x2]
        
        # Convert to PIL Image
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        cropped_regions.append({
            'id': i + 1,
            'image': cropped_pil,
            'bbox': box['bbox'],
            'confidence': box['confidence']
        })
    
    return cropped_regions