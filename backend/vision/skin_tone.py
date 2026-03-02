try:
    try:
        try:
            import cv2  # type: ignore
        except ImportError:
            import os
            import sys
            # Try common OpenCV install paths
            for path in [
                '/usr/local/lib/python3.9/site-packages',
                '/usr/lib/python3/dist-packages',
                os.path.join(sys.prefix, 'lib', 'site-packages')
            ]:
                if os.path.isdir(path) and path not in sys.path:
                    sys.path.insert(0, path)
            import cv2  # type: ignore
    except ImportError:
        import os
        import sys
        # Try common OpenCV install paths
        for path in [
            '/usr/local/lib/python3.9/site-packages',
            '/usr/lib/python3/dist-packages',
            os.path.join(sys.prefix, 'lib', 'site-packages')
        ]:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        try:
            import cv2  # type: ignore
        except ImportError:
            raise ImportError(
                "OpenCV (cv2) could not be resolved. "
                "Please install it via: pip install opencv-python"
            )
except ImportError:
    import sys
    sys.path.append('/usr/local/lib/python3.9/site-packages')  # Adjust path as needed
    pass  # cv2 import is handled in the try block above
try:
    try:
        try:
            import numpy as np  # type: ignore
        except ImportError:
            import os
            import sys
            # Try common NumPy install paths
            for path in [
                '/usr/local/lib/python3.9/site-packages',
                '/usr/lib/python3/dist-packages',
                os.path.join(sys.prefix, 'lib', 'site-packages')
            ]:
                if os.path.isdir(path) and path not in sys.path:
                    sys.path.insert(0, path)
            import numpy as np  # type: ignore
    except ImportError:
        import os
        import sys
        # Try common NumPy install paths
        for path in [
            '/usr/local/lib/python3.9/site-packages',
            '/usr/lib/python3/dist-packages',
            os.path.join(sys.prefix, 'lib', 'site-packages')
        ]:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        import numpy as np  # type: ignore
except ImportError:
    import os
    import sys
    # Try common NumPy install paths
    for path in [
        '/usr/local/lib/python3.9/site-packages',
        '/usr/lib/python3/dist-packages',
        os.path.join(sys.prefix, 'lib', 'site-packages')
    ]:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        import numpy as np  # type: ignore
    except ImportError:
        raise ImportError(
            "NumPy could not be resolved. "
            "Please install it via: pip install numpy"
        )

def detect_skin_tone(image_path):
    """
    Detects the skin tone from an image.

    Args:
        image_path: The path to the image file.

    Returns:
        A string representing the detected skin tone.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Perform k-means clustering to find the dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the center colors to integer values
    centers = np.uint8(centers)

    # Find the most frequent color
    dominant_color = centers[np.argmax(np.unique(labels, return_counts=True)[1])]

    # Convert the dominant color to HSV
    hsv_color = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0]

    # Define the range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Check if the dominant color is within the skin color range
    if lower_skin[0] <= hsv_color[0] <= upper_skin[0] and \
       lower_skin[1] <= hsv_color[1] <= upper_skin[1] and \
       lower_skin[2] <= hsv_color[2] <= upper_skin[2]:
        
        # Classify the skin tone based on the hue
        if hsv_color[0] < 10:
            return "warm"
        elif hsv_color[0] > 170:
            return "cool"
        else:
            return "neutral"
    else:
        return "unknown"


