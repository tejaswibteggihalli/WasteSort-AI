"""
test_accuracy.py
-----------------
Automated batch accuracy evaluation for WasteSort AI (EfficientNet-B0).

Point it at a folder of labelled test images — filenames like plastic1.jpg,
plastic_07.png, Glass3.jpeg, METAL-2.webp (class name + optional separator +
digits, case-insensitive) — and it will run every image through the model,
compare predictions to the filename-derived ground truth, and print overall
accuracy, per-class accuracy, a confusion matrix, and a list of
misclassified images (also saved to CSV).

Usage:
    python test_accuracy.py --test_dir ./test_images --model_path best_model.pth
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225]),
])


def build_model():
    """Same architecture as in app.py — must match exactly for state_dict to load."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(CLASSES)),
    )
    return model


def load_model(model_path):
    model = build_model()
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def extract_label(filename):
    """'plastic_12.jpg' -> 'plastic'. Returns None if no class name matches."""
    stem = os.path.splitext(filename)[0].lower()
    stem = re.sub(r'[\d_\-\s]+$', '', stem)  # strip trailing digits/separators
    if stem in CLASSES:
        return stem
    for c in CLASSES:
        if stem.startswith(c):
            return c
    return None


@torch.no_grad()
def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    tensor = val_tfms(image).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])


def main():
    parser = argparse.ArgumentParser(description="Evaluate WasteSort AI accuracy on a labelled test folder.")
    parser.add_argument(
        "--test_dir",
        default=r"C:\Users\tejas\Desktop\WasteSort-AI\dataset",
        help="Folder containing labelled test images.",
    )
    parser.add_argument("--model_path", default="best_model.pth", help="Path to trained model weights (.pth).")
    parser.add_argument("--save_csv", default="misclassified.csv", help="Where to save misclassified results.")
    args = parser.parse_args()

    if not os.path.isdir(args.test_dir):
        sys.exit(f"Test directory not found: {args.test_dir}")
    if not os.path.isfile(args.model_path):
        sys.exit(f"Model file not found: {args.model_path}")

    print(f"Loading model from {args.model_path} on {DEVICE}...")
    model = load_model(args.model_path)

    files = sorted(f for f in os.listdir(args.test_dir) if f.lower().endswith(VALID_EXTS))
    if not files:
        sys.exit(f"No images found in {args.test_dir}")

    print(f"Found {len(files)} images. Running inference...\n")

    n_classes = len(CLASSES)
    class_idx = {c: i for i, c in enumerate(CLASSES)}
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    total = 0
    correct = 0
    skipped = []
    misclassified = []

    for fname in files:
        true_label = extract_label(fname)
        if true_label is None:
            skipped.append((fname, "couldn't parse class from filename"))
            continue

        path = os.path.join(args.test_dir, fname)
        try:
            pred_label, confidence = predict(path, model)
        except Exception as e:
            skipped.append((fname, f"error reading/predicting: {e}"))
            continue

        total += 1
        conf_matrix[class_idx[true_label]][class_idx[pred_label]] += 1

        if pred_label == true_label:
            correct += 1
        else:
            misclassified.append((fname, true_label, pred_label, confidence))

    if total == 0:
        sys.exit("No valid labelled images were processed — check your filenames/extensions.")

    overall_acc = correct / total * 100

    print("=" * 50)
    print(f"OVERALL ACCURACY: {correct}/{total} = {overall_acc:.2f}%")
    print("=" * 50)
    print(f"{'Class':<12}{'Correct':<10}{'Total':<10}{'Accuracy':<10}")
    for c in CLASSES:
        i = class_idx[c]
        class_total = int(conf_matrix[i].sum())
        class_correct = int(conf_matrix[i][i])
        if class_total:
            print(f"{c:<12}{class_correct:<10}{class_total:<10}{class_correct/class_total*100:.2f}%")
        else:
            print(f"{c:<12}{0:<10}{0:<10}n/a")

    print("\nConfusion matrix (rows = true, cols = predicted)")
    print("true\\pred".ljust(12) + "".join(c[:6].ljust(8) for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(c.ljust(12) + "".join(str(conf_matrix[i][j]).ljust(8) for j in range(n_classes)))

    if misclassified:
        print(f"\n{len(misclassified)} misclassified image(s):")
        for fname, true_l, pred_l, conf in misclassified:
            print(f"  {fname}: true={true_l}  pred={pred_l}  (conf={conf*100:.1f}%)")

        with open(args.save_csv, "w") as f:
            f.write("filename,true_label,predicted_label,confidence\n")
            for fname, true_l, pred_l, conf in misclassified:
                f.write(f"{fname},{true_l},{pred_l},{conf:.4f}\n")
        print(f"\nSaved misclassified details to {args.save_csv}")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for fname, reason in skipped:
            print(f"  {fname}: {reason}")


if __name__ == "__main__":
    main()
