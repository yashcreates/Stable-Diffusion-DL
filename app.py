import streamlit as st
import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch
import time
import io
import concurrent.futures

# Set the device
DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"

# Load the model and tokenizer
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
model_file = "./data/Inkpunk-Diffusion-v2.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Function to generate the image using the model
def generate_image(prompt, uncond_prompt, input_image, strength, do_cfg, cfg_scale, sampler, num_inference_steps, seed):
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    return Image.fromarray(output_image)

# Streamlit UI
st.title("AI Image Generator")
st.write("Generate an image from a text prompt using a model made by YAP.")

# Input prompt from user
prompt = st.text_input("Enter your image prompt", "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.")

# Parameters for the model (you can expose these to the UI if needed)
uncond_prompt = ""
do_cfg = True
cfg_scale = 8
input_image = None
strength = 0.9
sampler = "ddpm"
num_inference_steps = 50
seed = 42

# Total time simulation (20 minutes = 1200 seconds)
total_time = 1200  # 20 minutes in seconds

# Function to manage progress bar and image generation concurrently
def generate_with_progress_bar():
    progress_bar = st.progress(0)
    time_remaining_text = st.empty()

    # Start the image generation in a background thread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(generate_image, prompt, uncond_prompt, input_image, strength, do_cfg, cfg_scale, sampler, num_inference_steps, seed)

        total_steps = 100  # Assuming we update the bar in 100 steps
        step_time = total_time / total_steps  # Time per step

        for i in range(total_steps):
            # Check if image generation is complete
            if future.done():
                break

            # Update the progress bar
            progress_bar.progress(i + 1)

            # Update time remaining
            time_remaining = total_time - ((i + 1) * step_time)
            minutes, seconds = divmod(time_remaining, 60)
            time_remaining_text.text(f"Time remaining: {int(minutes)} minutes {int(seconds)} seconds")

            # Simulate work being done
            time.sleep(0.1)  # Adjust this to match actual processing time

        # Wait for the image generation to complete
        output_image = future.result()

    return output_image

# Start generating button
if st.button("Generate Image"):
    st.write("Generating image...")

    # Run the progress bar and image generation concurrently
    output_image = generate_with_progress_bar()

    # Once the image generation is complete, display the image
    st.image(output_image, caption="Generated Image", use_column_width=True)

    # Allow users to download the image
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(label="Download Image", data=byte_im, file_name="generated_image.png", mime="image/png")
