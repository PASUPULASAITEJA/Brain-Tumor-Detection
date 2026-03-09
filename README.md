# Brain Tumor Detection

A deep learning application for detecting brain tumors from MRI images using Convolutional Neural Networks (CNN).

## Project Structure

```
brain-tumor-detection/
├── dataset/
│   ├── yes/          # MRI images with tumors
│   └── no/           # MRI images without tumors
├── model/
│   └── brain_tumor_model.h5
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your MRI images to the dataset folders:
   - Place images with tumors in `dataset/yes/`
   - Place images without tumors in `dataset/no/`

3. Train the model:
```bash
python train_model.py
```

4. Run the web application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Features

- CNN-based brain tumor detection
- Web interface for image upload
- Real-time prediction with confidence scores
- Model training with data augmentation

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Flask 2.3+

## Usage

Upload an MRI image through the web interface to get a prediction on whether a brain tumor is present.
