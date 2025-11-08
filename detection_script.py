#!/usr/bin/env python3
"""
Gloved vs Bare Hand Detection — Assessment Format
--------------------------------------------------

Detects:
  - gloved_hand
  - bare_hand

For every image:
  - Saves annotated image → out_dir/<filename>_annotated.jpg
  - Saves per-image JSON log → out_dir/logs/<filename>.json

Each JSON is in this format:
{
  "filename": "image1.jpg",
  "detections": [
    {"label": "gloved_hand", "confidence": 0.92, "bbox": [x1, y1, x2, y2]},
    {"label": "bare_hand", "confidence": 0.85, "bbox": [x1, y1, x2, y2]}
  ]
}
"""

import argparse
import json
from pathlib import Path
import cv2
from ultralytics import YOLO


# ---------------- Argument Parser (Your Version) ----------------
def parse_args():
    p = argparse.ArgumentParser(description="YOLO detection on images (no MediaPipe)")
    p.add_argument("--model", default=r"./best.pt", type=str, help="Path to YOLO model weights (.pt)")
    p.add_argument("--source", default=r"./input", type=str, help="Path to image or directory")
    p.add_argument("--out_dir", default=r"./output", type=str, help="Output directory")
    p.add_argument("--conf", default=0.3, type=float, help="YOLO confidence threshold (default 0.5)")
    p.add_argument("--iou", default=0.45, type=float, help="YOLO NMS IoU threshold")
    p.add_argument("--device", default=None, type=str, help="Device: 'cpu', '0', etc.")
    p.add_argument("--imgsz", default=640, type=int, help="YOLO inference size")
    p.add_argument("--line_thickness", default=2, type=int, help="Box thickness")
    return p.parse_args()


# ---------------- Utility Functions ----------------
def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def draw_box(img, xyxy, label, conf, thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(0, y1 - th - 4)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)


# ---------------- Main Function ----------------
def main():
    args = parse_args()

    input_path = Path(args.source)
    output_dir = ensure_dir(args.out_dir)
    logs_dir = ensure_dir(output_dir / "logs")

    # Load YOLO model
    print(f"🔹 Loading model: {args.model}")
    model = YOLO(args.model)
    if args.device:
        model.to(args.device)

    # Collect images
    valid_exts = {".jpg", ".jpeg", ".png"}
    if input_path.is_file() and input_path.suffix.lower() in valid_exts:
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = [p for p in sorted(input_path.rglob("*")) if p.suffix.lower() in valid_exts]
    else:
        raise FileNotFoundError(f"No valid images found at {input_path}")

    print(f"📁 Found {len(image_paths)} image(s). Starting detection...")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        results = model.predict(
            source=img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )
        r = results[0]
        names = r.names
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]

        detections = []
        annotated = img.copy()

        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0].item() if hasattr(b.conf[0], "item") else b.conf[0])
                cls = int(b.cls[0].item() if hasattr(b.cls[0], "item") else b.cls[0])
                label = names[cls] if 0 <= cls < len(names) else str(cls)

                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [
                        round(float(xyxy[0]), 2),
                        round(float(xyxy[1]), 2),
                        round(float(xyxy[2]), 2),
                        round(float(xyxy[3]), 2)
                    ]
                })
                draw_box(annotated, xyxy, label, conf, args.line_thickness)

        # Save annotated image
        annotated_path = output_dir / f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        # Save JSON log for this image
        json_path = logs_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"filename": img_path.name, "detections": detections}, f, indent=2)

        print(f"✅ Processed {img_path.name} → {json_path.name}")

    print("\n🎯 Detection complete.")
    print(f"Annotated images → {output_dir}")
    print(f"Logs (per image) → {logs_dir}")


# ---------------- Entry Point ----------------
if __name__ == "__main__":
    main()
