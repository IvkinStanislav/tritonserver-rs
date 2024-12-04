# **Image Detection Model**  

## **Model Source**  
- **Base Model**: YOLOv8 (from the official Ultralytics library)  
- **Postprocessing & Ensemble Model**: [GitHub - YOLOv8 Triton Integration](https://github.com/omarabid59/yolov8-triton/tree/main)  

## **Setup Instructions**  
1. Install Ultralytics YOLO version 8.0.51:  
   ```sh
   $ pip install ultralytics==8.0.51
   ```  
2. Export the model to ONNX format with dynamic shapes and opset 16:  
   ```sh
   $ yolo export model=yolov8n.pt format=onnx dynamic=True opset=16
   ```  
3. Move the exported model to the appropriate folder and clean up the `.pt` file:  
   ```sh
   $ rm yolov8n.pt
   $ mv yolov8n.onnx yolov8_onnx/1/model.onnx
   ```  

## **Class Labels**  
The class labels can be found in:  
- [../data/input/image_classes.csv](../data/input/image_classes.csv)  

To generate this CSV, you can use the following script:  
```python
from ultralytics import YOLO
import pandas as pd

model = YOLO('/content/yolov8n.pt')

classes = [{'class_id': x, 'class_name': y} for x, y in model.names.items()]

pd.DataFrame(classes).to_csv('data/image_classes.csv', index=False)
```

---