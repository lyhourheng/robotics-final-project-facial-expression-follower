import cv2, onnxruntime as ort, numpy as np

def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression"""
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

session = ort.InferenceSession("yolov8n-face.onnx")
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    
    img = cv2.resize(frame, (320, 320))
    blob = img.transpose(2,0,1)[None].astype(np.float32)/255

    outputs = session.run(None, {input_name: blob})
    predictions = outputs[0][0].T  # Shape: [num_predictions, 5+num_classes]
    
    # Filter by confidence
    scores = predictions[:, 4]
    mask = scores > 0.5
    filtered = predictions[mask]
    
    if len(filtered) > 0:
        # Extract boxes and scores
        boxes = filtered[:, :4]
        scores = filtered[:, 4]
        
        # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Apply NMS
        keep_indices = nms(boxes_xyxy, scores)
        
        # Scale and draw boxes
        scale_x = w / 320
        scale_y = h / 320
        
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            score = scores[idx]
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("face", frame)
    if cv2.waitKey(1)==27:
        break
