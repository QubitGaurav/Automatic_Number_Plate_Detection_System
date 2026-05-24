# KnightSight ANPR Pipeline: Edge-Optimized Vehicle Intelligence

KnightSight is a modular, edge-optimized Automatic Number Plate Recognition (ANPR) and Vehicle Intelligence Pipeline. It is designed to perform end-to-end processing, including vehicle detection, license plate localization, and OCR-based text extraction (specifically tailored for Indian license plates), with a focus on high performance in challenging real-world conditions.

## Features

- **End-to-End Pipeline**: Seamlessly integrates vehicle detection, plate localization, and OCR.
- **Edge-Optimized**: Designed to run efficiently using lightweight models (like YOLOv8n) suitable for edge devices.
- **Modular Architecture**: Separate components for vehicle detection, plate detection, and the ANPR OCR engine, making it easy to swap models or update specific parts of the pipeline.
- **Interactive Dashboard**: Includes a Streamlit-based web interface to easily upload images, run inference, and visualize bounding boxes and recognized text.
- **Custom Training Ready**: Contains scripts for converting JSON dataset annotations to YOLO format and fine-tuning YOLOv8 for custom license plate detection.

```

## Setup & Installation

1. **Clone or Navigate to the Project Directory**
   Ensure you are in the `ANRP` directory.

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   The project requires PyTorch, Ultralytics YOLOv8, OpenCV, Streamlit, and OCR engines.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you plan to use Tesseract for OCR, ensure the Tesseract system binary is installed on your OS (e.g., `sudo apt-get install tesseract-ocr` on Ubuntu).*

## Usage

### 1. Running the Interactive Dashboard

The easiest way to test the pipeline is via the Streamlit dashboard:

```bash
streamlit run app.py
```

- Open the provided local URL in your browser (usually `http://localhost:8501`).
- Upload an image containing vehicles.
- The pipeline will run and display the original image alongside the annotated result with vehicle bounding boxes, license plate boxes, and extracted text. The structured JSON output is also displayed.

### 2. Using the Pipeline via Code

You can integrate the `VehicleIntelligencePipeline` directly into your own Python scripts:

```python
import cv2
from pipeline import VehicleIntelligencePipeline

# Initialize the pipeline
pipeline = VehicleIntelligencePipeline()

# Load an image
image = cv2.imread('path/to/your/image.jpg')

# Run inference
# Returns: structured results list, raw vehicle data, raw plate data
results, vehicles, plates = pipeline.process_image(image_array=image)

# Print extracted plate text
for res in results:
    print(f"Detected Plate: {res['plate_text']} (Confidence: {res['ocr_confidence']:.2f})")

# Annotate and save the image
annotated_img = pipeline.annotate_image(image, results, vehicles)
cv2.imwrite('output.jpg', annotated_img)
```

## Data Preparation and Training

To improve the accuracy of license plate detection, fine-tune YOLO11 on the JSON labels in `Dataset/labels`.

1. **Prepare the Python environment**
   Recommended on Ubuntu 22.04:
   ```bash
   sudo apt update
   sudo apt install -y python3.11 python3.11-venv build-essential
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements-gpu.txt
   ```

2. **Convert the dataset annotations**
   ```bash
   python scripts/convert_json_to_yolo.py
   ```

3. **Configure `data.yaml`**
   The repository already uses a portable `path: ./Dataset` setup. Verify the dataset still contains:
   - `Dataset/images`
   - `Dataset/labels`

4. **Train the model**
   For a GTX 1650, start with `yolo11n.pt` and a conservative batch size:
   ```bash
   python scripts/train_yolo11.py --epochs 50 --batch 8 --imgsz 640 --device 0
   ```

   If you want to test the training flow without GPU, use:
   ```bash
   python scripts/train_yolo11.py --epochs 1 --batch 2 --device cpu
   ```

   The best checkpoint is saved to `runs/license_plate_detector/weights/best.pt` and the app will automatically pick it up through `models/plate_detector.py`.

## License & Acknowledgements
Built using Ultralytics YOLO11, Streamlit, and EasyOCR.

## Dataset
https://drive.google.com/file/d/1WpOjFnPfj-tfmHrUaJ7ypQLUsFqrJhAx/view?usp=sharing
