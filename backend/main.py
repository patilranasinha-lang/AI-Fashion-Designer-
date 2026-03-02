from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from vision.body_shape import detect_body_shape
from vision.skin_tone import detect_skin_tone
from vision.face_shape import detect_face_shape
from fashion.recommender import recommend_outfits
from rendering.segmentation import segment_person
from rendering.warping import warp_garment
from rendering.diffusion import render_tryon_image
import shutil
import io
import cv2
import mediapipe as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Fashion Stylist API"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Save the uploaded file
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Analyze the image
    body_shape = detect_body_shape(file.filename)
    skin_tone = detect_skin_tone(file.filename)
    face_shape = detect_face_shape(file.filename)

    return {
        "body_shape": body_shape,
        "skin_tone": skin_tone,
        "face_shape": face_shape,
    }

@app.post("/recommend")
async def get_recommendations(body_shape: str, skin_tone: str, occasion: str):
    return recommend_outfits(body_shape, skin_tone, occasion)

@app.post("/render")
async def render_image(person_file: UploadFile = File(...), garment_file: UploadFile = File(...)):
    # Save the uploaded files
    with open(person_file.filename, "wb") as buffer:
        shutil.copyfileobj(person_file.file, buffer)
    with open(garment_file.filename, "wb") as buffer:
        shutil.copyfileobj(garment_file.file, buffer)

    # Get pose landmarks
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    person_image = cv2.imread(person_file.filename)
    image_rgb = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return {"error": "Could not find pose landmarks in the image."}
    pose_landmarks = results.pose_landmarks.landmark

    # Render the try-on image
    person_image = cv2.imread(person_file.filename)
    garment_image = cv2.imread(garment_file.filename)
    segmented_person, mask = segment_person(person_file.filename)
    warped_garment = warp_garment(person_image, garment_image, pose_landmarks)
    rendered_image = render_tryon_image(segmented_person, warped_garment, mask)

    # Return the rendered image
    img_byte_arr = io.BytesIO()
    rendered_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")
