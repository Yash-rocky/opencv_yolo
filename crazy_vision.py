import cv2
import time
from ultralytics import YOLO
from deepface import DeepFace

# --- 1. PRODUCT SPECIFICATION DATABASE ---
# A dictionary mockup mimicking an inventory database
PRODUCT_DATABASE = {
    "cell phone": {
        "Name": "Flagship Smartphone X",
        "Display": "6.7-inch OLED, 120Hz",
        "Processor": "NextGen Octa-core AI Chip",
        "RAM/Storage": "12GB RAM / 256GB NVMe",
        "Camera": "108MP Triple-Lens System"
    },
    "laptop": {
        "Name": "UltraBook Pro 14",
        "Display": "14.2-inch Liquid Retina",
        "Processor": "M-Series Ultra 12-Core",
        "RAM/Storage": "16GB Unified / 512GB SSD",
        "Battery": "Up to 18 hours real-world"
    }
}

def main():
    # Load YOLOv8 Model (Fastest option)
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    
    # Timing configuration to avoid overloading DeepFace on every single frame
    last_emotion_check = 0
    emotion_interval = 0.5  # Analyze emotion every 0.5 seconds
    current_emotion = "Analyzing..."

    print("System active. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        h, w, _ = frame.shape

        # --- STEP A: OBJECT DETECTION & SPECS LOOKUP ---
        results = model(frame, conf=0.5, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label_name = model.names[cls_id]

                # Draw the standard box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_name.upper(), (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # TRIGGER: Check if the object is in our tech database
                if label_name in PRODUCT_DATABASE:
                    specs = PRODUCT_DATABASE[label_name]
                    
                    # Create a slick transparent overlay panel on the right side of the screen
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w - 320, 10), (w - 10, 200), (40, 40, 40), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                    # Print out the tech specs
                    cv2.putText(frame, f"DETECTED: {specs['Name']}", (w - 310, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    y_offset = 65
                    for key, val in specs.items():
                        if key == "Name": continue
                        text = f"- {key}: {val}"
                        cv2.putText(frame, text, (w - 310, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        y_offset += 25

                # --- STEP B: EMOTION PROCESSING (IF A PERSON IS SEEN) ---
                if label_name == "person":
                    if current_time - last_emotion_check > emotion_interval:
                        try:
                            # Crop face area slightly loose to help DeepFace read expressions
                            face_crop = frame[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
                            if face_crop.size > 0:
                                analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
                                current_emotion = analysis[0]['dominant_emotion']
                        except Exception:
                            pass
                        last_emotion_check = current_time
                    
                    # Display the current mood tag above the person's head
                    cv2.putText(frame, f"Mood: {current_emotion.upper()}", (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Show Output
        cv2.imshow("Sci-Fi HUD: Emotion & Tech Specs", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()