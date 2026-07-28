"""
Experiment 8: Image Generation Application Using Diffusion Models

Aim:
To generate images from text prompts using the Stable Diffusion
v1.5 model from Hugging Face.
"""

from diffusers import StableDiffusionPipeline
import torch


def load_pipeline() -> StableDiffusionPipeline:
    """
    Load the Stable Diffusion model.

    Returns:
        StableDiffusionPipeline object.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    print(f"Using device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    )

    pipe = pipe.to(device)

    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    output_file: str = "generated_city.png",
) -> None:
    """
    Generate an image from a text prompt.

    Args:
        pipe: Stable Diffusion pipeline.
        prompt: Image generation prompt.
        output_file: Output image filename.
    """

    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    image.save(output_file)

    print("\nImage generated successfully!")
    print(f"Saved as: {output_file}")


def main() -> None:
    print("=" * 60)
    print("STABLE DIFFUSION IMAGE GENERATION")
    print("=" * 60)

    pipe = load_pipeline()

    prompt = (
        "A futuristic city skyline at sunset, "
        "digital art, highly detailed"
    )

    print("\nPrompt:")
    print(prompt)

    generate_image(
        pipe,
        prompt,
    )


if __name__ == "__main__":
    main()
