import os
import glob
import json
import base64
import argparse
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import timm
from torchvision import transforms
from openai import OpenAI

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

class UltrasoundAnalyzer:
    def __init__(
        self,
        yolo_weights: str = "yolo11n.pt", # Placeholder path
        convnext_weights: str = "convnextv2_tiny.fcmae", # Placeholder model name/path
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "google/gemini-3-flash-preview", # Example model
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Initialize YOLO v11
        print(f"Loading YOLO model from {yolo_weights}...")
        # In a real scenario, ensure 'yolo11n.pt' or specific weights exist
        self.yolo_model = YOLO(yolo_weights) 

        # 2. Initialize ConvNeXt v2
        print(f"Loading ConvNeXt model ({convnext_weights})...")
        # Assuming a custom trained model for planes, but here loading a pretrained one for demo
        self.classifier = timm.create_model(
            "convnext_tiny.dinov3_lvd1689m",
            pretrained=False, 
            num_classes=6,
            checkpoint_path=convnext_weights
        )
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # Preprocessing for ConvNeXt
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Mapping index to anatomical plane name
        self.plane_labels = {
            0: "Abdominal Circumference View",
            1: "Amniotic Fluid",
            2: "Cardiac View",
            3: "Femur View",
            4: "Renal View",
            5: "Thalamic View"
        }

        # 3. Initialize OpenAI client for OpenRouter
        self.or_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.or_api_key:
            print("Warning: OPENROUTER_API_KEY not found. VLM extraction might fail.")
        
        self.vlm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.or_api_key,
        )
        self.vlm_model = openrouter_model

    def detect_and_crop(self, image_path: str) -> List[np.ndarray]:
        """
        Detects measurement indicators using YOLO and returns cropped images of them.
        """
        # Run inference
        results = self.yolo_model(image_path)
        crops = []
        original_img = cv2.imread(image_path)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # Crop
                crop = original_img[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
        
        return crops

    def classify_plane(self, image_path: str) -> str:
        """
        Classifies the anatomical plane of the ultrasound image using ConvNeXt v2.
        """
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.classifier(input_tensor)
            if output.shape[1] > 1:
                probs = torch.softmax(output, dim=1)
                max_prob, pred_idx = torch.max(probs, dim=1)
                max_prob_val = max_prob.item()
                pred_idx_val = pred_idx.item()
                if max_prob_val < 0.6:
                    return "Unknown"
                return self.plane_labels.get(pred_idx_val, "Unknown")
            else:
                return "FeatureVector (Classifier not fine-tuned)"

    def extract_metrics_with_vlm(self, crop_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Uses a VLM to read text/numbers from the cropped indicator images.
        """
        extracted_data = {}
        
        for i, crop in enumerate(crop_images):
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', crop)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            prompt = (
                "Identify the measurement label and value in this ultrasound indicator crop. "
                "Return JSON format with keys 'label' and 'value'. "
                "Example: {'label': 'BPD', 'value': '45.2 mm'}"
            )

            try:
                response = self.vlm_client.chat.completions.create(
                    model=self.vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                content = response.choices[0].message.content
                # Simple parsing attempt - in production use structured outputs or better parsing
                extracted_data[f"crop_{i}"] = content
            except Exception as e:
                print(f"Error calling VLM: {e}")
                extracted_data[f"crop_{i}"] = "Error extracting data"

        return extracted_data

    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        results = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in image_extensions:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        print(f"Found {len(files)} images in {folder_path}")

        for img_path in files:
            print(f"Processing {img_path}...")
            
            # Step 1 & 2: Detect and Crop
            crops = self.detect_and_crop(img_path)
            
            # Step 3: Extract metrics
            metrics = self.extract_metrics_with_vlm(crops)
            
            # Step 4: Classify Plane
            plane = self.classify_plane(img_path)
            
            results.append({
                "filename": os.path.basename(img_path),
                "anatomical_plane": plane,
                "measurements": metrics
            })
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Generate Obstetrics Ultrasound Report")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder containing ultrasound images")
    parser.add_argument("--yolo", type=str, default="/home/null/best.pt", help="Path to YOLO weights")
    parser.add_argument("--convnext", type=str, default="/home/null/timmconvnext_tiny_dinov3_lvd1689m/model.safetensors", help="ConvNeXt model name or path")
    
    args = parser.parse_args()
    
    # 1. run analysis pipeline
    analyzer = UltrasoundAnalyzer(
        yolo_weights=args.yolo,
        convnext_weights=args.convnext,
        openrouter_api_key="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",
    )
    
    analysis_results = analyzer.process_folder(args.input_dir)
    
    # Convert results to string for RLM
    context_str = json.dumps(analysis_results, indent=2)
    print("\nAggregated Context for RLM:")
    print(context_str)
    
    # 2. RLM Report Generation
    print("\nInitializing RLM for Report Generation...")
    
    # Setup logger
    logger = RLMLogger(log_dir="./logs")
    
    # Initialize RLM (using OpenAI as per quickstart, assumes OPENAI_API_KEY is set for this part)
    # The user asked to use RLM for the final generation.
    rlm = RLM(
        backend="openrouter",
        backend_kwargs={
            "model_name": "x-ai/grok-4.1-fast",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        environment="local",
        environment_kwargs={},
        max_depth=3,
        logger=logger,
        verbose=True
    )
    
    examples = str(json.load(open("/home/null/rlm/examples/example.json")))
    
    prompt = (
        f"You are an expert obstetrician assistant. "
        f"Based on the following structured data extracted from ultrasound images, "
        f"generate a comprehensive and professional obstetrics ultrasound report. "
        # f"Highlight any missing standard planes if applicable. "
        f"The final report should consist of Finding and Impression in Chinese. "
        f"Here are some examples of the report: {examples}"
        f"\n\nData:\n{context_str}"
    )
    
    print("\nGenerating Report...")
    report = rlm.completion(prompt)
    
    print("\n" + "="*50)
    print("GENERATED REPORT")
    print("="*50)
    print(report.response)
    print("="*50)

if __name__ == "__main__":
    main()
