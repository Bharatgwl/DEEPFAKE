# SDXL DreamBooth Studio

A complete Streamlit application for training custom SDXL models with DreamBooth and generating images.

## Features

- 📤 **Upload & Prepare Training Data**: Web interface for uploading images
- 🔧 **Automated Training Setup**: Generates ready-to-use Google Colab code
- 🎨 **Image Generation**: Generate images with your trained models
- ☁️ **Hugging Face Integration**: Models automatically uploaded to HF Hub
- 💾 **Google Drive Support**: Optional backup to Drive

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Hugging Face

1. Create account at [huggingface.co](https://huggingface.co)
2. Get your token from Settings → Access Tokens
3. Create `.env` file:
```env
HF_TOKEN=your_token_here
```

### 3. (Optional) Configure Google Drive

1. Create Google Cloud project
2. Enable Drive API
3. Download OAuth credentials as `credentials.json`

### 4. Run the App
```bash
streamlit run app.py
```

## Usage

### Training a Model

1. Go to **Upload & Train** page
2. Configure your model (trigger word, name)
3. Upload 10-20 training images
4. Click "Prepare & Generate Colab Code"
5. Copy the generated code to Google Colab
6. Run the Colab notebook (45-60 min)
7. Model automatically uploads to Hugging Face

### Generating Images

1. Go to **Generate** page
2. Enter your HF model ID (username/model-name)
3. Load the model
4. Enter prompt and generate!

## Requirements

- **For Training**: Google Colab (free GPU)
- **For Generation**: NVIDIA GPU with CUDA
- **Storage**: Hugging Face account (free)

## Tips

- Use 10-20 varied images for best results
- Include trigger word in all prompts
- Adjust LoRA strength for different effects
- Use negative prompts to avoid unwanted elements

## Troubleshooting

**"No GPU detected"**: The generation page requires a CUDA GPU. Use Colab for generation if you don't have one locally.

**Drive upload fails**: Make sure `credentials.json` is configured correctly.

**Model not found**: Check that the model ID is correct and the model is public (or you're logged in).

## License

MIT
