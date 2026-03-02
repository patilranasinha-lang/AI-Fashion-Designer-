try:
    import torch  # type: ignore
except ImportError as e:
    raise RuntimeError("torch is not installed. Please `pip install torch`.") from e
try:
    from diffusers import StableDiffusionInpaintPipeline  # type: ignore
except ImportError as e:
    raise RuntimeError("diffusers is not installed. Please `pip install diffusers transformers accelerate`.") from e
try:
    from PIL import Image  # type: ignore
except ImportError as e:
    raise RuntimeError("Pillow is not installed. Please `pip install pillow`.") from e


def render_tryon_image(person_image, warped_garment, mask):
    """
    Renders the final try-on image using a diffusion model.

    Args:
        person_image: The segmented image of the person.
        warped_garment: The warped garment image.
        mask: The segmentation mask of the person.

    Returns:
        The final, rendered try-on image.
    """
    # Load the Stable Diffusion inpainting pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=dtype)
    pipe = pipe.to(device)

    # Prepare the images for the pipeline
    person_image_pil = Image.fromarray(person_image)
    warped_garment_pil = Image.fromarray(warped_garment)
    mask_pil = Image.fromarray(mask).convert("L")

    # Generate the image
    image = pipe(prompt="a photo of a person wearing a garment", image=person_image_pil, mask_image=mask_pil, guidance_scale=7.5).images[0]

    return image

