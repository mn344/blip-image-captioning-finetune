# BLIP Image Captioning Fine-Tuning

Fine-tuning [Salesforce's BLIP](https://github.com/salesforce/BLIP) (`blip-image-captioning-base`) model on a custom image–caption dataset to generate captions tailored to a specific domain, trained and run on Google Colab.

## Overview

This project fine-tunes the pretrained BLIP image captioning model using a custom dataset of images and their corresponding captions. The workflow covers the full pipeline:

1. **Data preparation** – Extracts images from zipped archives and loads captions from a tab-separated text file.
2. **Dataset splitting** – Splits the data into train/validation sets (90/10).
3. **Fine-tuning** – Trains the BLIP model on the custom dataset using Hugging Face's `Trainer` API.
4. **Model export** – Saves the fine-tuned model and processor, then zips them for storage on Google Drive.
5. **Inference** – Reloads the fine-tuned model and generates captions for new, user-uploaded images.

## Tech Stack

- Python
- [Transformers](https://github.com/huggingface/transformers) (Hugging Face)
- PyTorch
- PIL (Pillow)
- Google Colab (GPU runtime)

## Model

- **Base model:** `Salesforce/blip-image-captioning-base`
- **Task:** Image captioning (conditional text generation from image input)

## Dataset Format

Captions are provided in a tab-separated `.txt` file, one entry per line:

```
image_filename.jpg<TAB>A short caption describing the image.
```

Images referenced in the captions file must exist in the extracted image directory; entries with missing images or empty captions are automatically filtered out.

## Project Structure (Colab-based)

```
/content/all_images/          # Extracted training images
/content/captions_final.txt   # Captions file (image -> caption)
/content/train.json           # Training split
/content/val.json             # Validation split
/content/drive/MyDrive/blip_finetuned_v3_checkpoints/   # Model checkpoints & final model
```

## Usage

### 1. Setup
Install dependencies:
```bash
pip install -q transformers accelerate pillow
```

### 2. Prepare Data
Mount Google Drive, extract image zip archives, and load the captions file into `train.json` / `val.json`.

### 3. Fine-Tune
Load the base BLIP processor and model, wrap the dataset in a custom PyTorch `Dataset`, and train using Hugging Face's `Trainer`.

### 4. Save & Export
The fine-tuned model and processor are saved locally and zipped for storage/download from Google Drive.

### 5. Run Inference
Reload the fine-tuned model and generate a caption for any uploaded image:
```python
inputs = processor(images=image, return_tensors="pt").to(device)
output_ids = model.generate(**inputs, max_new_tokens=64, num_beams=5)
caption = processor.decode(output_ids[0], skip_special_tokens=True)
```

## Notes

- Training and inference were run on Google Colab with GPU acceleration (`nvidia-smi` used to confirm GPU availability).
- The fine-tuned model checkpoints are stored on Google Drive rather than in this repository due to file size.

## License

Specify a license (e.g., MIT) if you'd like others to freely use or modify this code.
