try:
    import cv2  # type: ignore
except ImportError as e:
    raise RuntimeError("opencv-python is not installed. Please run `pip install opencv-python`.") from e
try:
    import numpy as np  # type: ignore
except ImportError:
    import sys
    print("Error: numpy is not installed. Please install it using 'pip install numpy'")
    sys.exit(1)
# numpy is already imported in the try/except block above; no need to re-import
try:
    import mediapipe as mp  # type: ignore
except ImportError as e:
    raise RuntimeError("mediapipe is not installed. Please `pip install mediapipe`.") from e


def warp_garment(person_image, garment_image, pose_landmarks):
    """
    Warps the garment image to fit the person's body.

    Args:
        person_image: The image of the person.
        garment_image: The image of the garment.
        pose_landmarks: The pose landmarks of the person.

    Returns:
        The warped garment image.
    """
    # Get the height and width of the images
    person_height, person_width, _ = person_image.shape
    garment_height, garment_width, _ = garment_image.shape

    # Define the source and destination points for the perspective transform
    src_pts = np.float32([[0, 0], [garment_width, 0], [garment_width, garment_height], [0, garment_height]])
    
    # Get the coordinates of the shoulders and hips
    left_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

    # Define the destination points based on the pose landmarks
    dst_pts = np.float32([
        [left_shoulder.x * person_width, left_shoulder.y * person_height],
        [right_shoulder.x * person_width, right_shoulder.y * person_height],
        [right_hip.x * person_width, right_hip.y * person_height],
        [left_hip.x * person_width, left_hip.y * person_height]
    ])

    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the garment image
    warped_garment = cv2.warpPerspective(garment_image, matrix, (person_width, person_height))

    return warped_garment
