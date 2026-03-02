import cv2
import mediapipe as mp

def detect_body_shape(image_path):
    """
    Detects the body shape from an image.

    Args:
        image_path: The path to the image file.

    Returns:
        A string representing the detected body shape.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return "unknown"

    landmarks = results.pose_landmarks.landmark

    # Get the coordinates of the shoulders, waist, and hips
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_waist = landmarks[mp_pose.PoseLandmark.LEFT_WAIST]
    right_waist = landmarks[mp_pose.PoseLandmark.RIGHT_WAIST]

    # Calculate the width of the shoulders, waist, and hips
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    waist_width = abs(left_waist.x - right_waist.x)
    hip_width = abs(left_hip.x - right_hip.x)

    # Classify the body shape based on the measurements
    if shoulder_width * 1.05 > hip_width and shoulder_width * 0.95 < hip_width:
        if waist_width < hip_width * 0.75 and waist_width < shoulder_width * 0.75:
            return "hourglass"
        else:
            return "rectangle"
    elif shoulder_width > hip_width * 1.05:
        return "inverted triangle"
    elif hip_width > shoulder_width * 1.05:
        if waist_width < hip_width * 0.75:
            return "pear"
        else:
            return "triangle"
    else:
        return "rectangle"


