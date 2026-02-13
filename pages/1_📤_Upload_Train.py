import streamlit as st
import os
import zipfile
from datetime import datetime
import json
from dotenv import load_dotenv

# Try to import drive upload (optional)
try:
    from utils.drive_upload import upload_to_drive
    DRIVE_AVAILABLE = True
except:
    DRIVE_AVAILABLE = False
    st.warning("Google Drive upload not configured. Files will be saved locally only.")

load_dotenv()

st.set_page_config(page_title="Upload & Train", page_icon="📤", layout="wide")

st.title("📤 Upload Training Images")
st.markdown("Create your custom SDXL DreamBooth LoRA model")

# Configuration
st.header("1️⃣ Model Configuration")

col1, col2 = st.columns(2)

with col1:
    trigger_word = st.text_input(
        "Trigger Word",
        "sksPerson",
        help="Unique word to trigger your subject (e.g., 'jdoe', 'mylogo', 'painting_style')"
    )
    
    hf_username = st.text_input(
        "Hugging Face Username",
        "",
        help="Your HF username (model will be saved as username/model-name)"
    )

with col2:
    class_name = st.selectbox(
        "Class Type",
        ["person", "object", "style", "character"],
        help="What type of subject are you training?"
    )
    
    model_name = st.text_input(
        "Model Name",
        f"sdxl-{trigger_word}-lora",
        help="Name for your model on Hugging Face Hub"
    )

# Image upload
st.header("2️⃣ Upload Images")

st.info("""
📸 **Image Guidelines:**
- Upload 10-20 high-quality images
- Different angles, expressions, lighting
- Clear, well-lit photos work best
- Avoid duplicates or very similar images
""")

uploaded_files = st.file_uploader(
    "Select images (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Drag and drop or click to upload"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} images uploaded")
    
    # Preview images
    if st.checkbox("Preview images"):
        cols = st.columns(5)
        for idx, file in enumerate(uploaded_files[:10]):  # Show first 10
            with cols[idx % 5]:
                st.image(file, caption=file.name, use_container_width=True)
        
        if len(uploaded_files) > 10:
            st.caption(f"... and {len(uploaded_files) - 10} more images")

# Training configuration
st.header("3️⃣ Training Settings (Optional)")

with st.expander("Advanced Settings"):
    col1, col2 = st.columns(2)
    
    with col1:
        max_train_steps = st.slider("Training Steps", 400, 1500, 800, 100)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 5e-4],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}"
        )
    
    with col2:
        resolution = st.selectbox("Resolution", [512, 768, 1024], index=2)
        batch_size = st.selectbox("Batch Size", [1, 2], index=0)

# Prepare for training
st.header("4️⃣ Prepare Training Data")

if st.button("🚀 Prepare & Generate Colab Code", type="primary", disabled=len(uploaded_files) < 10):
    if len(uploaded_files) < 10:
        st.error("❌ Please upload at least 10 images")
        st.stop()
    
    if not hf_username:
        st.error("❌ Please enter your Hugging Face username")
        st.stop()
    
    # Create data directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/training_{timestamp}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save uploaded images
    for file in uploaded_files:
        with open(f"{data_dir}/{file.name}", "wb") as f:
            f.write(file.getbuffer())
    
    # Create zip file
    zip_filename = f"dreambooth_images_{timestamp}.zip"
    zip_path = f"data/{zip_filename}"
    
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for img in os.listdir(data_dir):
            zipf.write(f"{data_dir}/{img}", img)
    
    st.success(f"✅ Created zip file: {zip_filename}")
    
    # Save configuration
    config = {
        "trigger_word": trigger_word,
        "class_name": class_name,
        "model_name": model_name,
        "hf_username": hf_username,
        "num_images": len(uploaded_files),
        "max_train_steps": max_train_steps,
        "learning_rate": str(learning_rate),
        "resolution": resolution,
        "timestamp": timestamp
    }
    
    config_path = f"data/config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Upload to Google Drive (optional)
    if DRIVE_AVAILABLE:
        with st.spinner("Uploading to Google Drive..."):
            try:
                folder_id = upload_to_drive(zip_path, os.getenv("GOOGLE_DRIVE_FOLDER", "sdxl_dreambooth_inputs"))
                st.success("✅ Uploaded to Google Drive!")
            except Exception as e:
                st.warning(f"⚠️ Could not upload to Drive: {e}")
                st.info("You can manually upload the zip file below")
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        with open(zip_path, "rb") as f:
            st.download_button(
                "📥 Download Training Images",
                f,
                file_name=zip_filename,
                mime="application/zip"
            )
    
    with col2:
        with open(config_path, "rb") as f:
            st.download_button(
                "📥 Download Config",
                f,
                file_name=f"config_{timestamp}.json",
                mime="application/json"
            )
    
    # Generate Colab code
    st.markdown("---")
    st.header("5️⃣ Google Colab Training Code")
    
    instance_prompt = f"a photo of {trigger_word} {class_name}"
    repo_id = f"{hf_username}/{model_name}"
    
    colab_code = f'''# SDXL DreamBooth Training
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# =====================================
# STEP 1: Mount Google Drive
# =====================================
from google.colab import drive
drive.mount("/content/drive")

# =====================================
# STEP 2: Install Dependencies
# =====================================
!pip install -q diffusers["training"] transformers accelerate peft safetensors

# =====================================
# STEP 3: Extract Training Images
# =====================================
import os, zipfile

# Option 1: If you uploaded to Google Drive
ZIP_PATH = "/content/drive/MyDrive/sdxl_dreambooth_inputs/{zip_filename}"

# Option 2: If you're uploading directly to Colab
# from google.colab import files
# uploaded = files.upload()  # Upload your zip file
# ZIP_PATH = list(uploaded.keys())[0]

IMAGE_DIR = "/content/training_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(IMAGE_DIR)

print(f"✅ Extracted {{len(os.listdir(IMAGE_DIR))}} images")

# =====================================
# STEP 4: Login to Hugging Face
# =====================================
from huggingface_hub import login

# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN = "your_huggingface_token_here"  # ⚠️ REPLACE THIS
login(token=HF_TOKEN)

!git clone https://github.com/huggingface/diffusers.git
# =====================================
# STEP 5: Clone Diffusers & Train
# =====================================
!git clone https://github.com/huggingface/diffusers.git

# =====================================
# STEP 6: Training Configuration
# =====================================
INSTANCE_PROMPT = "a photo of bharat person"
OUTPUT_DIR = "/content/sdxl_lora"

RESOLUTION = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 1e-4
MAX_STEPS = 800

# =====================================
# STEP 7: Start Training (WORKING COMMAND)
# =====================================
!accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="/content/training_images" \
  --output_dir="/content/sdxl_lora" \
  --instance_prompt="a photo of bharat person" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --checkpointing_steps=200 \
  --mixed_precision="fp16" \
  --seed=42

print("✅ Training finished")

# =====================================
# STEP 6: Upload to Hugging Face Hub
# =====================================
from huggingface_hub import HfApi, create_repo

REPO_ID = "{repo_id}"

# Create repository
create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
print(f"✅ Created repo: {{REPO_ID}}")

# Create model card
model_card = """---
license: creativeml-openrail-m
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags:
  - stable-diffusion-xl
  - stable-diffusion-xl-diffusers
  - text-to-image
  - diffusers
  - lora
  - dreambooth
instance_prompt: {instance_prompt}
---

# SDXL DreamBooth LoRA - {model_name}

This is a LoRA (Low-Rank Adaptation) model trained with DreamBooth on Stable Diffusion XL.

## Trigger Word
`{trigger_word}`

## Usage
```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("{repo_id}")

prompt = "{instance_prompt}, professional portrait, studio lighting"
image = pipe(prompt, num_inference_steps=30).images[0]
image.save("output.png")
```

## Training Details
- **Base Model:** stabilityai/stable-diffusion-xl-base-1.0
- **Training Steps:** {max_train_steps}
- **Learning Rate:** {learning_rate}
- **Resolution:** {resolution}x{resolution}
- **Batch Size:** {batch_size}
- **Instance Prompt:** {instance_prompt}
- **Images:** {len(uploaded_files)}
"""

with open("/content/sdxl_lora/README.md", "w") as f:
    f.write(model_card)

# Upload model
api = HfApi()
api.upload_folder(
    folder_path="/content/sdxl_lora",
    repo_id=REPO_ID,
    repo_type="model"
)

print(f"🎉 Model uploaded successfully!")
print(f"View at: https://huggingface.co/{{REPO_ID}}")

# =====================================
# STEP 7: Test Generation
# =====================================
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights(REPO_ID)

test_prompts = [
    "{instance_prompt}, professional headshot, 8k",
    "{instance_prompt}, casual outdoor photo, natural lighting",
    "{instance_prompt}, artistic portrait, dramatic lighting"
]

print("Generating test images...")
for i, prompt in enumerate(test_prompts):
    image = pipe(prompt, num_inference_steps=30).images[0]
    image.save(f"/content/test_{{i+1}}.png")
    display(image)

print("✅ All done! Check your model at https://huggingface.co/{repo_id}")
'''
    
    st.code(colab_code, language="python")
    
    # Download Colab notebook
    st.download_button(
        "📥 Download Colab Code (.py)",
        colab_code,
        file_name=f"train_sdxl_{timestamp}.py",
        mime="text/x-python"
    )
    
    # Instructions
    st.markdown("---")
    st.success(f"""
    ### ✅ Next Steps:
    
    1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)
    2. **Change Runtime to GPU**: Runtime → Change runtime type → T4 GPU
    3. **Copy the code above** into a new notebook
    4. **Replace `your_huggingface_token_here`** with your actual HF token
    5. **Run all cells** (this will take 45-60 minutes)
    6. **Your model will be at**: `https://huggingface.co/{repo_id}`
    
    After training, go to the **Generate** page to create images!
    """)
    
    # Model card preview
    with st.expander("📄 Model Card Preview"):
        st.markdown(f"""
        ## {model_name}
        
        **Trigger Word:** `{trigger_word}`  
        **Class:** `{class_name}`  
        **Repository:** `{repo_id}`  
        **Training Images:** {len(uploaded_files)}  
        **Steps:** {max_train_steps}  
        **Resolution:** {resolution}x{resolution}
        """)

elif uploaded_files and len(uploaded_files) < 10:
    st.warning(f"⚠️ Upload at least 10 images (currently: {len(uploaded_files)})")