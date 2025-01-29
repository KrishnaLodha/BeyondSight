import cv2
from deepface import DeepFace

def recognize_face(frame):
    try:
        # Analyze face in the frame
        result = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        
        if result:
            # Extract details
            age = result[0]['age']
            gender = result[0]['dominant_gender']
            emotion = result[0]['dominant_emotion']

            return f"Age: {age}, Gender: {gender}, Emotion: {emotion}"
    except Exception as e:
        print(f"Error in face recognition: {e}")
    
    return "No Face Detected"
