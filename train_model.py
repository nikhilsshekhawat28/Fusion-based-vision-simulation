from ultralytics import YOLO
import torch

# ✅ Step 1: Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ✅ Step 2: Load YOLOv8 Model (pretrained)
model = YOLO('yolov8n.pt').to(device)

# ✅ Step 3: Train on the custom underwater dataset
results = model.train(
    data='./dataset/data.yaml',       # Path to dataset configuration
    epochs=50,                        # Number of epochs
    imgsz=640,                        # Image size (use 416 or 640 for faster training)
    batch=8,                          # Batch size (reduce if using CPU)
    device=device,                     # Use GPU if available
    workers=2,                        # Use 2 workers for data loading
    project='./output',               # Output directory
    name='underwater_training'        # Run name
)

# ✅ Step 4: Evaluate the Model
metrics = model.val()

# ✅ Step 5: Export and Save the Model
model.export(format='onnx')  # Export model to ONNX format (optional)

# ✅ Save final model weights
model.save('./output/weights/best.pt')
print("Model saved successfully!")
