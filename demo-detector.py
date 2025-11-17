import cv2, onnxruntime as ort, numpy as np

session = ort.InferenceSession("yolov8n-face.onnx")
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (320,320))
    blob = img.transpose(2,0,1)[None].astype(np.float32)/255

    outputs = session.run(None, {input_name: blob})
    detections = outputs[0][0].T  # Transpose to get [num_detections, 5+num_classes]
    
    # Scale coordinates back to original frame size
    scale_x = frame.shape[1] / 320
    scale_y = frame.shape[0] / 320

    for d in detections:
        x1, y1, x2, y2, score = d[0], d[1], d[2], d[3], d[4]
        if score > 0.5:
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("face", frame)
    if cv2.waitKey(1)==27:
        break
