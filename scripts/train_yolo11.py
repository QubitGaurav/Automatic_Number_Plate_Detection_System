import os
import json
import glob
import argparse
import torch
from ultralytics import YOLO
from PIL import Image

def convert_json_to_yolo(dataset_path):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    
    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        print(f"Warning: Ensure {images_dir} and {labels_dir} exist.")
        return
        
    print("Checking for required JSON to YOLO label conversion...")
    converted_count = 0
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.json'):
            continue
            
        json_path = os.path.join(labels_dir, filename)
        txt_path = os.path.join(labels_dir, filename.replace('.json', '.txt'))
        
        if os.path.exists(txt_path):
            continue
            
        base_name = filename.replace('.json', '')
        image_files = glob.glob(os.path.join(images_dir, f"{base_name}.*"))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            continue
            
        image_path = image_files[0]
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            continue
            
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except:
                continue
                
        yolo_lines = []
        for obj in data:
            class_id = 0
            x_min = obj['x']
            y_min = obj['y']
            width = obj['width']
            height = obj['height']
            
            x_center = x_min + (width / 2.0)
            y_center = y_min + (height / 2.0)
            
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height
            
            yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
            
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        converted_count += 1
        
    if converted_count > 0:
        print(f"Successfully converted {converted_count} JSON labels to YOLO .txt format.")
    else:
        print("No new JSON labels needed conversion.")

def train(epochs=1, subset=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    dataset_dir = os.path.join(project_root, 'Dataset')
    data_yaml_path = os.path.join(project_root, 'data.yaml')
    runs_dir = os.path.join(project_root, 'runs')

    convert_json_to_yolo(dataset_dir)

    print(f"Starting Highly Optimized YOLO11 training using: {data_yaml_path} for {epochs} epochs")

    model = YOLO("yolo11s.pt")

    # Decide device automatically
    device_arg = 0 if torch.cuda.is_available() else 'cpu'
    if device_arg == 0:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = 'cuda:0'
        print(f"Using GPU: {device_name}")
    else:
        print("No CUDA device detected — falling back to CPU")

    train_kwargs = dict(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=8,
        workers=4,
        cache=False,
        device=device_arg,
        optimizer="AdamW",
        project="runs",
        name="license_plate_detector",
        patience=20,
    )

    if device_arg == 'cpu':
        train_kwargs['batch'] = 2
        train_kwargs['workers'] = 0

    try:
        model.train(**train_kwargs)
    except Exception as e:
        err_str = str(e)
        print("Training failed with error:\n", err_str)
        if 'cuda' in err_str.lower() or 'cudnn' in err_str.lower() or 'acceleratorerror' in err_str.lower() or 'unknown error' in err_str.lower():
            print("Detected CUDA-related error. Attempting recovery: reduce batch and retry...")
            try:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if train_kwargs.get('batch', 0) > 1:
                    train_kwargs['batch'] = max(1, train_kwargs['batch'] // 2)
                model.train(**train_kwargs)
            except Exception:
                print("Retry with smaller batch also failed. Falling back to CPU training.")
                train_kwargs['device'] = 'cpu'
                train_kwargs['batch'] = 2
                train_kwargs['workers'] = 0
                model.train(**train_kwargs)
        else:
            raise

    print("Training complete. Model saved in runs/license_plate_detector/weights/best.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO11 on license plate dataset")
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    args = parser.parse_args()
    
    train(epochs=args.epochs)
