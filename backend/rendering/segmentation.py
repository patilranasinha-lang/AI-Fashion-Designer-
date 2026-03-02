try:
    import cv2  # type: ignore
except ImportError as e:
    raise RuntimeError("opencv-python is not installed. Please `pip install opencv-python`.") from e
try:
    import numpy as np  # type: ignore
except ImportError as e:
    raise RuntimeError("numpy is not installed. Please `pip install numpy`.") from e
try:
    import torch  # type: ignore
except ImportError as e:
    raise RuntimeError("torch is not installed. Please `pip install torch`.") from e
try:
    import torchvision.transforms as T  # type: ignore
except ImportError as e:
    raise RuntimeError("torchvision is not installed. Please `pip install torchvision`.") from e


def segment_person(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True)
    model.to(device).eval()
    image = cv2.imread(image_path)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    output_predictions = output.argmax(0)
    mask = (output_predictions == 15).to(torch.uint8).cpu().numpy()
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image, mask
