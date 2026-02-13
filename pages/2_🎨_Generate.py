import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import io
from datetime import datetime
import os

st.set_page_config(page_title="Generate Images", page_icon="🎨", layout="wide")

st.title("🎨 Generate Images with Your Model")

# Check for CUDA
if not torch.cuda.is_available():
    st.error("""
    ❌ **No GPU detected!** 
    
    This page requires a CUDA-enabled GPU to run. You have two options:
    
    1. **Use Google Colab** (Recommended - Free GPU):
       - Open a new Colab notebook
       - Enable GPU runtime
       - Use the generation code from your training notebook
    
    2. **Run locally** if you have an NVIDIA GPU:
       - Install CUDA toolkit
       - Install PyTorch with CUDA support
       - Restart this app
    """)
    st.stop()

# Initialize session state
if 'pipe' not in st.session_state:
    st.session_state.pipe = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

# Model loading
st.header("1️⃣ Load Model")

col1, col2 = st.columns([2, 1])

with col1:
    model_id = st.text_input(
        "Hugging Face Model ID",
        placeholder="username/model-name",
        help="Enter the full repo ID (e.g., 'john/sdxl-portrait-lora')"
    )

with col2:
    lora_scale = st.slider(
        "LoRA Strength",
        0.0, 2.0, 1.0, 0.1,
        help="How strongly to apply the LoRA. 1.0 = normal, >1.0 = stronger"
    )

load_col1, load_col2 = st.columns([1, 3])

with load_col1:
    load_button = st.button("🔄 Load Model", type="primary", use_container_width=True)

with load_col2:
    if st.session_state.current_model:
        st.success(f"✅ Loaded: {st.session_state.current_model}")

if load_button and model_id:
    try:
        with st.spinner(f"Loading {model_id}... (this may take 1-2 minutes)"):
            # Load base SDXL model
            if st.session_state.pipe is None:
                st.session_state.pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                ).to("cuda")
                
                # Enable optimizations
                st.session_state.pipe.enable_attention_slicing()
            
            # Load LoRA weights
            st.session_state.pipe.load_lora_weights(model_id)
            st.session_state.current_model = model_id
            
        st.success(f"✅ Model loaded: {model_id}")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Make sure the model ID is correct and the model is public or you're logged in.")

# # Generation interface
# if st.session_state.pipe and st.session_state.current_model:
#     st.markdown("---")
#     st.header("2️⃣ Generate Images")
    
#     # Prompt input
#     prompt = st.text_area(
#         "Prompt",
#         height=100,
#         placeholder="a photo of sksPerson person, professional portrait, studio lighting, 8k, high detail",
#         help="Describe what you want to generate. Include your trigger word!"
#     )
    
#     negative_prompt = st.text_area(
#         "Negative Prompt (Optional)",
#         height=80,
#         value="blurry, low quality, distorted, deformed, ugly, bad anatomy",
#         help="What to avoid in the generation"
#     )
    
#     # Advanced settings
#     with st.expander("⚙️ Advanced Settings"):
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             num_inference_steps = st.slider("Steps", 10, 100, 30, 5)
#             guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        
#         with col2:
#             width = st.selectbox("Width", [512, 768], index=1)
#             height = st.selectbox("Height", [512, 768], index=1)
        
#         with col3:
#             num_images = st.slider("Number of Images", 1, 4, 1)
#             seed = st.number_input("Seed (-1 for random)", -1, 999999, -1)
    
#     # Generate button
#     if st.button("🎨 Generate", type="primary", use_container_width=True):
#         if not prompt:
#             st.error("Please enter a prompt")
#             st.stop()
        
#         try:
#             # Set seed
#             if seed != -1:
#                 generator = torch.Generator(device="cuda").manual_seed(seed)
#             else:
#                 generator = None
            
#             with st.spinner(f"Generating {num_images} image(s)... (~{num_inference_steps * 0.3:.0f}s)"):
#                 # Generate images
#                 result = st.session_state.pipe(
#                     prompt=prompt,
#                     negative_prompt=negative_prompt if negative_prompt else None,
#                     num_inference_steps=num_inference_steps,
#                     guidance_scale=guidance_scale,
#                     width=width,
#                     height=height,
#                     num_images_per_prompt=num_images,
#                     generator=generator,
#                     cross_attention_kwargs={"scale": lora_scale}
#                 )
                
#                 images = result.images
                
#                 # Save to session state
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 for idx, img in enumerate(images):
#                     st.session_state.generated_images.append({
#                         "image": img,
#                         "prompt": prompt,
#                         "timestamp": timestamp,
#                         "index": idx,
#                         "seed": seed if seed != -1 else "random"
#                     })
            
#             st.success(f"✅ Generated {num_images} image(s)!")
            
#         except Exception as e:
#             st.error(f"❌ Generation error: {str(e)}")
    
#     # Display generated images
#     if st.session_state.generated_images:
#         st.markdown("---")
#         st.header("3️⃣ Generated Images")
        
#         # Show images in grid
#         cols_per_row = 2
#         for i in range(0, len(st.session_state.generated_images), cols_per_row):
#             cols = st.columns(cols_per_row)
            
#             for j, col in enumerate(cols):
#                 idx = i + j
#                 if idx < len(st.session_state.generated_images):
#                     img_data = st.session_state.generated_images[idx]
                    
#                     with col:
#                         st.image(img_data["image"], use_container_width=True)
                        
#                         # Image info
#                         with st.expander(f"Image {idx + 1} Details"):
#                             st.text(f"Prompt: {img_data['prompt'][:100]}...")
#                             st.text(f"Timestamp: {img_data['timestamp']}")
#                             st.text(f"Seed: {img_data['seed']}")
                        
#                         # Download button
#                         buf = io.BytesIO()
#                         img_data["image"].save(buf, format="PNG")
#                         st.download_button(
#                             "📥 Download",
#                             buf.getvalue(),
#                             file_name=f"generated_{img_data['timestamp']}_{idx}.png",
#                             mime="image/png",
#                             use_container_width=True
#                         )
        
#         # Clear button
#         if st.button("🗑️ Clear All Images"):
#             st.session_state.generated_images = []
#             st.rerun()
# =====================================
# Generation interface
# =====================================
if st.session_state.pipe and st.session_state.current_model:
    st.markdown("---")
    st.header("2️⃣ Generate Images")

    # Prompt input
    prompt = st.text_area(
        "Prompt",
        height=100,
        placeholder="a photo of sksPerson person, professional portrait, studio lighting, 8k, high detail",
        help="Describe what you want to generate. Include your trigger word!"
    )

    negative_prompt = st.text_area(
        "Negative Prompt (Optional)",
        height=80,
        value="blurry, low quality, distorted, deformed, ugly, bad anatomy",
        help="What to avoid in the generation"
    )

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2, col3 = st.columns(3)

        with col1:
            num_inference_steps = st.slider("Steps", 10, 50, 30, 5)
            guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 7.5, 0.5)

        with col2:
            # ✅ FIXED index bug + RTX 3050 safe
            width = st.selectbox("Width", [512, 768], index=1)
            height = st.selectbox("Height", [512, 768], index=1)

        with col3:
            num_images = st.slider("Number of Images", 1, 2, 1)
            seed = st.number_input("Seed (-1 for random)", -1, 999999, -1)

    # Generate button
    if st.button("🎨 Generate", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt")
            st.stop()

        try:
            # ===============================
            # GPU optimizations (SAFE)
            # ===============================
            st.session_state.pipe.enable_attention_slicing()
            st.session_state.pipe.enable_vae_slicing()

            # xFormers (optional, safe)
            try:
                st.session_state.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers not installed → ignore safely

            # ===============================
            # Seed handling
            # ===============================
            generator = None
            if seed != -1:
                generator = torch.Generator(device="cuda").manual_seed(seed)

            # ===============================
            # Generate
            # ===============================
            with st.spinner("Generating image..."):
                with torch.inference_mode():
                    result = st.session_state.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        num_images_per_prompt=num_images,
                        generator=generator,
                        cross_attention_kwargs={"scale": lora_scale}
                    )

            images = result.images

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for idx, img in enumerate(images):
                st.session_state.generated_images.append({
                    "image": img,
                    "prompt": prompt,
                    "timestamp": timestamp,
                    "index": idx,
                    "seed": seed if seed != -1 else "random"
                })

            st.success("✅ Image generated successfully!")

        except torch.cuda.OutOfMemoryError:
            st.error("❌ CUDA Out of Memory! Try 512×512 or fewer steps.")
            torch.cuda.empty_cache()

        except Exception as e:
            st.error(f"❌ Generation error: {str(e)}")
else:
    st.info("👆 Load a model first to start generating images")
    
    # Example models
    st.markdown("---")
    st.header("Example Models to Try")
    
    st.markdown("""
    Don't have a model yet? Try these public LoRA models:
    
    - `artificialguybr/StudioGhibli.Redmond-StdGBRRedmAF-v2` - Studio Ghibli style
    - `nerijs/pixel-art-xl` - Pixel art style
    - `TheLastBen/Papercut_SDXL` - Papercut art style
    
    Or train your own using the **Upload & Train** page!
    """)

# Sidebar - Quick tips
with st.sidebar:
    st.header("💡 Tips")
    
    st.markdown("""
    **Good Prompts:**
    - Include your trigger word
    - Be specific and detailed
    - Mention style, lighting, quality
    
    **Example:**
```
    a photo of sksPerson person,
    professional headshot,
    studio lighting, sharp focus,
    8k uhd, high quality
```
    
    **Negative Prompts:**
    - Remove unwanted elements
    - Fix common issues
    
    **Settings:**
    - More steps = better quality (slower)
    - Higher guidance = follows prompt more
    - LoRA strength: adjust effect intensity
    """)
    
    st.markdown("---")
    st.markdown("**GPU Memory:**")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        st.text(f"Used: {allocated:.2f} GB")
        st.text(f"Reserved: {reserved:.2f} GB")