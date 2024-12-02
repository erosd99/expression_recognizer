import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from typing import NamedTuple
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import time

class Face(NamedTuple):
    bbox: tuple[int, int, int, int]
    confidence: float
    emotion: str = "unknown"
    emotion_score: float = 0.0

def prepare_face_batch(faces: list[np.ndarray], batch_size=8) -> np.ndarray:
    """Prepare a batch of faces for emotion recognition"""
    processed_faces = []
    for face_img in faces:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype(np.float32) / 255.0
        rgb = np.stack([normalized] * 3, axis=-1)
        processed_faces.append(rgb)
    
    # Pad batch if needed
    while len(processed_faces) % batch_size != 0:
        processed_faces.append(np.zeros((48, 48, 3), dtype=np.float32))
    
    return np.array(processed_faces)

def process_faces_batch(face_imgs: list[np.ndarray], emotion_model: tf.keras.Model) -> list[tuple[str, float]]:
    """Process a batch of faces at once"""
    if not face_imgs:
        return []
    
    batch = prepare_face_batch(face_imgs)
    predictions = emotion_model.predict(batch, verbose=0)
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    results = []
    for pred in predictions[:len(face_imgs)]:  # Ignore padded predictions
        emotion_idx = np.argmax(pred)
        emotion_score = float(pred[emotion_idx])
        results.append((emotions[emotion_idx], emotion_score))
    
    return results

def run_detection(video_path: str, model_path: str, target_width=800, target_height=600):
    # Configure TensorFlow
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(6)
    
    # Load model
    emotion_model = tf.keras.models.load_model(model_path)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get original video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate scaling factor if original is larger than target
    should_resize = original_width > target_width or original_height > target_height
    if should_resize:
        # Calculate aspect ratio preserving scale
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale = min(width_ratio, height_ratio)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        print(f"Resizing from {original_width}x{original_height} to {new_width}x{new_height}")
    
    # Rest of initializations...
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    with ThreadPoolExecutor(max_workers=6) as face_pool:
        processing_times = deque(maxlen=30)
        
    
        try:
            while True:
                loop_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if should_resize:
                    frame = cv2.resize(frame, (new_width, new_height), 
                                     interpolation=cv2.INTER_AREA)
                
                # Process frame for face detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb_frame)
                
                faces = []
                if results.detections:
                    h, w, _ = frame.shape
                    face_images = []
                    face_metadata = []  # Store bbox and confidence
                    
                    # Collect all faces first
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = max(0, int(bbox.xmin * w))
                        y = max(0, int(bbox.ymin * h))
                        width = min(int(bbox.width * w), w - x)
                        height = min(int(bbox.height * h), h - y)
                        
                        face_img = frame[y:y+height, x:x+width]
                        if face_img.size == 0:
                            continue
                        
                        face_images.append(face_img)
                        face_metadata.append((x, y, width, height, detection.score[0]))
                    
                    # Process all faces in a single batch
                    emotions = process_faces_batch(face_images, emotion_model)
                    
                    # Combine results
                    for (x, y, w, h, conf), (emotion, score) in zip(face_metadata, emotions):
                        faces.append(Face(
                            bbox=(x, y, w, h),
                            confidence=conf,
                            emotion=emotion,
                            emotion_score=score
                        ))               
                # Draw results
                for face in faces:
                    x, y, w, h = face.bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = f"{face.emotion} ({face.emotion_score:.2f})"
                    cv2.putText(frame, text, (x, y + h + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate and display FPS
                processing_times.append(time.time() - loop_start)
                fps = 1.0 / (sum(processing_times) / len(processing_times))
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Face Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
if __name__ == "__main__":
    import pathlib
    data_path = pathlib.Path(__file__).parents[1] / "data"
    run_detection(data_path / "people_talking.mp4", data_path / "model.h5")
