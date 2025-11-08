# Glove vs No-Glove Detection

This project is part of a safety compliance system that checks whether workers are wearing gloves while working on factory floors.  
It uses a fine-tuned **YOLO model (Ultralytics)** to detect two classes in images:

- `glove`
- `no_glove`

The system takes a folder of images, runs object detection, and saves:
- Annotated images with bounding boxes.
- JSON logs for each image containing labels, confidence scores, and bounding boxes.

---

##  Dataset

I created a **custom dataset** by collecting and manually annotating images of hands with and without gloves using **Vott** in pascal voc format and then converted into yolo format.  
To improve model performance, I applied data augmentation techniques such as flips and zoom.  


**Classes:**
1. `no_glove`
2. `glove`

---

##  Model Details

- **Framework:** Ultralytics YOLO  
- **Base Model:** YOLOv8n   
- **Training Type:** Fine-tuned on a 2-class dataset  
- **Epochs:** 40  
 
## How to run my script
1. Install dependencies
**pip install ultralytics opencv-python**

2. Run detection
**python detection_script.py --source ./input --out_dir ./output --conf 0.3**

