import streamlit as st

st.set_page_config(
    page_title="SDXL DreamBooth Studio",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 SDXL DreamBooth Studio")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("📤 Train Your Model")
    st.markdown("""
    Upload 10-20 images of a person, object, or style to create your custom SDXL LoRA model.
    
    **Process:**
    1. Upload images
    2. Set trigger word
    3. Train in Google Colab (free GPU)
    4. Model saved to Hugging Face Hub
    
    **Time:** ~45-60 minutes
    """)
    
    if st.button("Start Training Setup →", use_container_width=True):
        st.switch_page("pages/1_📤_Upload_Train.py")

with col2:
    st.header("🎨 Generate Images")
    st.markdown("""
    Use your trained models to generate high-quality images.
    
    **Features:**
    - Load any Hugging Face LoRA model
    - Advanced prompt controls
    - Batch generation
    - Download results
    
    **Requirements:** Trained model on HF Hub
    """)
    
    if st.button("Generate Images →", use_container_width=True):
        st.switch_page("pages/2_🎨_Generate.py")

st.markdown("---")

st.info("""
💡 **How it works:**
1. Upload training images via this app
2. Train model in Google Colab (free GPU)
3. Model automatically uploads to Hugging Face
4. Generate unlimited images using your model
""")

# Quick stats
st.markdown("### 📊 Quick Info")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training Time", "45-60 min")
with col2:
    st.metric("Min Images", "10")
with col3:
    st.metric("Cost", "FREE")