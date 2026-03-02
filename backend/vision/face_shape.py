import cv2
import mediapipe as mp

def detect_face_shape(image_path):
    """
    Detects the face shape from an image.

    Args:
        image_path: The path to the image file.

    Returns:
        A string representing the detected face shape.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return "unknown"

    for face_landmarks in results.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        # Get the coordinates of the key facial landmarks
        face_width = abs(landmarks[454].x - landmarks[234].x)
        face_length = abs(landmarks[10].y - landmarks[152].y)
        jawline_width = abs(landmarks[172].x - landmarks[397].x)
        cheekbone_width = abs(landmarks[227].x - landmarks[447].x)


    # Classify the face shape based on the measurements
    if face_length > face_width:
        if cheekbone_width > jawline_width:
            return "heart"
        else:
            return "oblong"
    elif face_width > face_length:
        if cheekbone_width > jawline_width:
            return "round"
        else:
            return "square"
    else:
        if cheekbone_width > jawline_width:
            return "diamond"
        else:
            return "oval"


