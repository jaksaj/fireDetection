"""Visualisation script to run inference on multiple models and save results.

This script loads the best checkpoints from Iteration 1, 2, 3, 4, and 5,
runs inference on sample images, and saves visual results.
It can automatically find representative test images (Neither, Only_Fire,
Only_Smoke, Both, and Segmentation) from the dataset splits, or process
custom user-specified images.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import yaml

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt imports from src
try:
    from src.model import FireCNN, MobileNetV3FireClassifier
    from src.model_segmentation import LightweightUNet
    from src.dfire_labels import MULTICLASS_CLASS_NAMES, parse_yolo_label_file
except ImportError as err:
    print(f"Error importing modules from 'src': {err}")
    print("Please make sure you run this script from the project root directory.")
    sys.path.insert(0, str(Path.cwd()))
    from src.model import FireCNN, MobileNetV3FireClassifier
    from src.model_segmentation import LightweightUNet
    from src.dfire_labels import MULTICLASS_CLASS_NAMES, parse_yolo_label_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predictions across all iteration models and save visualized results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional directory containing images to test.",
    )
    parser.add_argument(
        "--image-paths",
        type=Path,
        nargs="+",
        default=None,
        help="Optional list of specific image paths to test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "visualizations",
        help="Directory where visualization results will be saved.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Path to the directory containing checkpoints.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of random samples to visualize when using auto-discovery.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Disable automatic selection of representative test images.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.warning("Config path %s not found. Using default parameters.", config_path)
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model_weights(checkpoint_path: Path, model: nn.Module, device: torch.device) -> bool:
    """Load classification/segmentation model weights from checkpoint, returning True if successful."""
    if not checkpoint_path.exists():
        logger.warning("Checkpoint not found at %s. Skipping this model.", checkpoint_path)
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get("epoch", "unknown")
            metrics = checkpoint.get("metrics", {})
            logger.info("Loaded checkpoint %s (epoch %s, metrics: %s)", checkpoint_path.name, epoch, metrics)
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded direct state dict from %s", checkpoint_path.name)
        model.to(device)
        model.eval()
        return True
    except Exception as e:
        logger.error("Error loading checkpoint %s: %s", checkpoint_path, e)
        return False


def load_iteration4_model(checkpoint_path: Path) -> Any:
    """Load YOLO26 model from checkpoint, returning YOLO model if successful."""
    if not checkpoint_path.exists():
        logger.warning("YOLO checkpoint not found at %s. Skipping Iteration 4.", checkpoint_path)
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(checkpoint_path))
        logger.info("Loaded YOLO checkpoint from %s", checkpoint_path)
        return model
    except ImportError:
        logger.error("Failed to import 'ultralytics' library. YOLO26 model cannot be evaluated.")
        logger.error("Please install it on your workstation: pip install ultralytics")
        return None
    except Exception as e:
        logger.error("Error loading YOLO model from %s: %s", checkpoint_path, e)
        return None


def get_representative_images(data_dir: Path, num_samples: int) -> list[dict[str, Any]]:
    """Scan D-Fire test split and collect one representative image of each multiclass category.

    Categories: Neither, Only_Fire, Only_Smoke, Both.
    """
    images_dir = data_dir / "test" / "images"
    labels_dir = data_dir / "test" / "labels"
    
    if not images_dir.is_dir() or not labels_dir.is_dir():
        logger.warning("Test dataset directories not found under %s. Cannot auto-select.", data_dir)
        return []

    # Map image paths by category
    categories: dict[str, list[Path]] = {
        "Neither": [],
        "Only_Fire": [],
        "Only_Smoke": [],
        "Both": []
    }
    
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            categories["Neither"].append(img_path)
            continue
            
        class_ids = parse_yolo_label_file(label_path)
        has_smoke = 0 in class_ids
        has_fire = 1 in class_ids
        
        if has_fire and has_smoke:
            categories["Both"].append(img_path)
        elif has_fire:
            categories["Only_Fire"].append(img_path)
        elif has_smoke:
            categories["Only_Smoke"].append(img_path)
        else:
            categories["Neither"].append(img_path)
            
    selected: list[dict[str, Any]] = []
    # Try to pick one image from each category
    for cat_name, paths in categories.items():
        if paths:
            chosen = random.choice(paths)
            selected.append({
                "path": chosen,
                "ground_truth": cat_name,
                "source": "D-Fire Test"
            })
            logger.info("Selected representative image for category '%s': %s", cat_name, chosen.name)
        else:
            logger.warning("No images found for category '%s'", cat_name)
            
    # Add random extra samples from COCO segmentation test split if available
    coco_test_dir = data_dir / "coco" / "test"
    if coco_test_dir.is_dir():
        coco_images = [p for p in coco_test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        # Filter out COCO JSON files
        coco_images = [p for p in coco_images if "_annotations" not in p.name]
        if coco_images:
            chosen = random.choice(coco_images)
            selected.append({
                "path": chosen,
                "ground_truth": "Unknown (COCO Segmentation)",
                "source": "COCO Test"
            })
            logger.info("Selected COCO segmentation image: %s", chosen.name)
            
    # If we didn't find enough, backfill with random test images
    all_images = categories["Neither"] + categories["Only_Fire"] + categories["Only_Smoke"] + categories["Both"]
    while len(selected) < num_samples and all_images:
        chosen = random.choice(all_images)
        if chosen not in [s["path"] for s in selected]:
            selected.append({
                "path": chosen,
                "ground_truth": "Random D-Fire",
                "source": "D-Fire Test"
            })
            
    return selected


def apply_overlay(image_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay segmentation mask onto the image.

    Class index mappings:
        0: Background (no overlay)
        1: Smoke (Orange/Yellow)
        2: Fire (Red)
    """
    overlay = image_np.copy()
    
    # Colors in RGB format
    smoke_color = np.array([255, 140, 0], dtype=np.uint8)  # DarkOrange
    fire_color = np.array([220, 20, 60], dtype=np.uint8)    # Crimson Red
    
    smoke_mask = (mask_np == 1)
    fire_mask = (mask_np == 2)
    
    overlay[smoke_mask] = (1 - alpha) * overlay[smoke_mask] + alpha * smoke_color
    overlay[fire_mask] = (1 - alpha) * overlay[fire_mask] + alpha * fire_color
    
    return overlay


def run_inference_on_image(
    img_path: Path,
    models: dict[str, Any],
    device: torch.device
) -> dict[str, Any]:
    """Preprocess image and run inference through all loaded models."""
    results = {}
    
    # Load original image
    with Image.open(img_path) as img:
        orig_img = img.convert("RGB")
    
    results["orig_img"] = orig_img
    
    # Preprocessing transforms
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    classification_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    segmentation_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    # Prepare batch inputs
    clf_tensor = classification_transform(orig_img).unsqueeze(0).to(device)
    seg_tensor = segmentation_transform(orig_img).unsqueeze(0).to(device)
    
    # Run Iteration 1 (Binary FireCNN)
    if "iteration1" in models:
        model = models["iteration1"]
        with torch.no_grad():
            logits = model(clf_tensor)
            prob = torch.sigmoid(logits).item()
        
        # Binary prediction
        if prob >= 0.5:
            pred_class = "Fire"
            conf = prob
        else:
            pred_class = "Normal"
            conf = 1.0 - prob
            
        results["iteration1"] = {"class": pred_class, "confidence": conf}
        
    # Run Iteration 2 (MobileNetV3 4-class)
    if "iteration2" in models:
        model = models["iteration2"]
        with torch.no_grad():
            logits = model(clf_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = torch.argmax(probs).item()
            conf = probs[pred_idx].item()
            
        pred_class = MULTICLASS_CLASS_NAMES[pred_idx]
        results["iteration2"] = {"class": pred_class, "confidence": conf}
        
    # Run Iteration 3 (Robust MobileNetV3 4-class)
    if "iteration3" in models:
        model = models["iteration3"]
        with torch.no_grad():
            logits = model(clf_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = torch.argmax(probs).item()
            conf = probs[pred_idx].item()
            
        pred_class = MULTICLASS_CLASS_NAMES[pred_idx]
        results["iteration3"] = {"class": pred_class, "confidence": conf}
        
    # Run Iteration 4 (YOLO26 Object Detection)
    if "iteration4" in models:
        model = models["iteration4"]
        try:
            # Run inference (disable verbose logging output)
            yolo_results = model(orig_img, verbose=False)[0]
            results["iteration4"] = {
                "boxes": yolo_results.boxes.xyxy.cpu().numpy(),
                "confs": yolo_results.boxes.conf.cpu().numpy(),
                "clss": yolo_results.boxes.cls.cpu().numpy(),
                "names": yolo_results.names
            }
        except Exception as e:
            logger.error("Error running YOLO inference: %s", e)

    # Run Iteration 5 (U-Net Semantic Segmentation)
    if "iteration5" in models:
        model = models["iteration5"]
        with torch.no_grad():
            logits = model(seg_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_mask = torch.argmax(probs, dim=0).cpu().numpy() # (256, 256)
            
        # Calculate percentages of fire/smoke pixels to log
        total_pixels = pred_mask.size
        smoke_pct = (pred_mask == 1).sum() / total_pixels * 100
        fire_pct = (pred_mask == 2).sum() / total_pixels * 100
        
        results["iteration5"] = {
            "mask": pred_mask,
            "smoke_pct": smoke_pct,
            "fire_pct": fire_pct
        }
        
    return results


def save_visualizations(
    img_name: str,
    results: dict[str, Any],
    ground_truth: str,
    output_dir: Path
) -> None:
    """Generate and save comparison grid and individual model outputs using Matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    orig_img = results["orig_img"]
    img_width, img_height = orig_img.size
    
    # 1. GENERATE COMPARISON GRID (2x3 Layout)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    # Helper to style text boxes based on prediction class
    def get_textbox_style(pred_class: str) -> dict[str, Any]:
        pred_lower = pred_class.lower()
        if "fire" in pred_lower or "both" in pred_lower:
            color = "#D32F2F"  # Red
        elif "smoke" in pred_lower:
            color = "#EF6C00"  # Orange
        else:
            color = "#2E7D32"  # Green
            
        return {
            "boxstyle": "round,pad=0.5",
            "facecolor": color,
            "alpha": 0.85,
            "edgecolor": "white",
            "linewidth": 1.5
        }

    # Font properties
    title_font = {"fontsize": 14, "weight": "bold", "color": "#1A237E"}
    label_font = {"fontsize": 13, "weight": "bold", "color": "white"}
    
    # Plot 1: Original Image
    ax = axes[0, 0]
    ax.imshow(orig_img)
    ax.set_title("Original Image", fontdict=title_font, pad=10)
    ax.axis("off")
    # Add Ground Truth Label
    ax.text(
        10, img_height - 15, 
        f"GT: {ground_truth}", 
        color="black", fontsize=10, weight="bold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="square,pad=0.3")
    )
    
    # Plot 2: Iteration 1 (Binary CNN)
    ax = axes[0, 1]
    ax.imshow(orig_img)
    ax.set_title("Iteration 1: Binary CNN", fontdict=title_font, pad=10)
    ax.axis("off")
    if "iteration1" in results:
        res = results["iteration1"]
        pred_txt = f"{res['class']}\nConf: {res['confidence']:.1%}"
        ax.text(
            img_width // 2, img_height // 2, pred_txt,
            ha="center", va="center", fontdict=label_font,
            bbox=get_textbox_style(res["class"])
        )
    else:
        ax.text(img_width // 2, img_height // 2, "Model N/A", ha="center", va="center", color="gray", fontsize=14)

    # Plot 3: Iteration 2 (MobileNetV3 Multiclass)
    ax = axes[0, 2]
    ax.imshow(orig_img)
    ax.set_title("Iteration 2: MobileNetV3 Multiclass", fontdict=title_font, pad=10)
    ax.axis("off")
    if "iteration2" in results:
        res = results["iteration2"]
        pred_txt = f"{res['class']}\nConf: {res['confidence']:.1%}"
        ax.text(
            img_width // 2, img_height // 2, pred_txt,
            ha="center", va="center", fontdict=label_font,
            bbox=get_textbox_style(res["class"])
        )
    else:
        ax.text(img_width // 2, img_height // 2, "Model N/A", ha="center", va="center", color="gray", fontsize=14)

    # Plot 4: Iteration 3 (Robust MobileNetV3)
    ax = axes[1, 0]
    ax.imshow(orig_img)
    ax.set_title("Iteration 3: Robust MobileNetV3", fontdict=title_font, pad=10)
    ax.axis("off")
    if "iteration3" in results:
        res = results["iteration3"]
        pred_txt = f"{res['class']}\nConf: {res['confidence']:.1%}"
        ax.text(
            img_width // 2, img_height // 2, pred_txt,
            ha="center", va="center", fontdict=label_font,
            bbox=get_textbox_style(res["class"])
        )
    else:
        ax.text(img_width // 2, img_height // 2, "Model N/A", ha="center", va="center", color="gray", fontsize=14)

    # Plot 5: Iteration 4 (YOLO26 Object Detection)
    ax = axes[1, 1]
    ax.imshow(orig_img)
    ax.set_title("Iteration 4: YOLO26 Object Detection", fontdict=title_font, pad=10)
    ax.axis("off")
    if "iteration4" in results:
        res = results["iteration4"]
        boxes = res["boxes"]
        confs = res["confs"]
        clss = res["clss"]
        names = res["names"]
        
        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            class_name = names.get(int(cls), f"class_{int(cls)}")
            
            # Select color based on class (fire = red, smoke = orange)
            color = "#DC143C" if "fire" in class_name.lower() else "#FF8C00"
            
            # Add bounding box
            rect = mpatches.Rectangle(
                (x1, y1), width, height,
                linewidth=2.5, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label text
            label_text = f"{class_name} {conf:.1%}"
            ax.text(
                x1, max(0, y1 - 6), label_text,
                color="white", weight="bold", fontsize=9,
                bbox=dict(facecolor=color, alpha=0.85, edgecolor='none', pad=2)
            )
    else:
        ax.text(img_width // 2, img_height // 2, "Model N/A", ha="center", va="center", color="gray", fontsize=14)

    # Plot 6: Iteration 5 (U-Net Segmentation Overlay)
    ax = axes[1, 2]
    if "iteration5" in results:
        res = results["iteration5"]
        # Resize original image to 256x256 to align with mask resolution
        resized_orig = orig_img.resize((256, 256))
        overlay_np = apply_overlay(np.array(resized_orig), res["mask"])
        ax.imshow(overlay_np)
        ax.set_title("Iteration 5: U-Net Segmentation", fontdict=title_font, pad=10)
        
        # Add legend patches
        smoke_patch = mpatches.Patch(color='#FF8C00', label=f'Smoke ({res["smoke_pct"]:.1f}%)')
        fire_patch = mpatches.Patch(color='#DC143C', label=f'Fire ({res["fire_pct"]:.1f}%)')
        ax.legend(handles=[fire_patch, smoke_patch], loc="upper right", fontsize=9, framealpha=0.9)
    else:
        ax.imshow(orig_img)
        ax.text(img_width // 2, img_height // 2, "Model N/A", ha="center", va="center", color="gray", fontsize=14)
        ax.set_title("Iteration 5: U-Net Segmentation", fontdict=title_font, pad=10)
    ax.axis("off")
    
    # Save Grid Fig
    grid_filename = output_dir / f"comparison_{img_name}.png"
    plt.savefig(grid_filename, bbox_inches="tight", dpi=180)
    plt.close()
    logger.info("Saved consolidated grid: %s", grid_filename)

    # 2. GENERATE AND SAVE INDIVIDUAL MODEL IMAGES (For slides layout flexibility)
    
    # Iteration 1
    if "iteration1" in results:
        res = results["iteration1"]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        ax.imshow(orig_img)
        pred_txt = f"Iteration 1 Binary CNN:\n{res['class']} ({res['confidence']:.1%})"
        ax.text(
            15, 35, pred_txt, fontdict=label_font,
            bbox=get_textbox_style(res["class"])
        )
        ax.axis("off")
        fn = output_dir / f"iter1_{img_name}.png"
        plt.savefig(fn, bbox_inches="tight")
        plt.close()

    # Iteration 2
    if "iteration2" in results:
        res = results["iteration2"]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        ax.imshow(orig_img)
        pred_txt = f"Iteration 2 Multiclass:\n{res['class']} ({res['confidence']:.1%})"
        ax.text(
            15, 35, pred_txt, fontdict=label_font,
            bbox=get_textbox_style(res["class"])
        )
        ax.axis("off")
        fn = output_dir / f"iter2_{img_name}.png"
        plt.savefig(fn, bbox_inches="tight")
        plt.close()

    # Iteration 3
    if "iteration3" in results:
        res = results["iteration3"]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        ax.imshow(orig_img)
        pred_txt = f"Iteration 3 Robust Multiclass:\n{res['class']} ({res['confidence']:.1%})"
        ax.text(
            15, 35, pred_txt, fontdict=label_font,
            bbox=get_textbox_style(res["class"])
        )
        ax.axis("off")
        fn = output_dir / f"iter3_{img_name}.png"
        plt.savefig(fn, bbox_inches="tight")
        plt.close()

    # Iteration 4
    if "iteration4" in results:
        res = results["iteration4"]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        ax.imshow(orig_img)
        
        boxes = res["boxes"]
        confs = res["confs"]
        clss = res["clss"]
        names = res["names"]
        
        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            class_name = names.get(int(cls), f"class_{int(cls)}")
            color = "#DC143C" if "fire" in class_name.lower() else "#FF8C00"
            
            rect = mpatches.Rectangle(
                (x1, y1), width, height,
                linewidth=2.5, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            label_text = f"{class_name} {conf:.1%}"
            ax.text(
                x1, max(0, y1 - 6), label_text,
                color="white", weight="bold", fontsize=9,
                bbox=dict(facecolor=color, alpha=0.85, edgecolor='none', pad=2)
            )
            
        ax.axis("off")
        fn = output_dir / f"iter4_{img_name}.png"
        plt.savefig(fn, bbox_inches="tight")
        plt.close()

    # Iteration 5
    if "iteration5" in results:
        res = results["iteration5"]
        # Raw Overlay
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        resized_orig = orig_img.resize((256, 256))
        overlay_np = apply_overlay(np.array(resized_orig), res["mask"])
        ax.imshow(overlay_np)
        
        # Add legend patches
        smoke_patch = mpatches.Patch(color='#FF8C00', label=f'Smoke ({res["smoke_pct"]:.1f}%)')
        fire_patch = mpatches.Patch(color='#DC143C', label=f'Fire ({res["fire_pct"]:.1f}%)')
        ax.legend(handles=[fire_patch, smoke_patch], loc="upper right", fontsize=10, framealpha=0.9)
        
        ax.axis("off")
        fn = output_dir / f"iter5_{img_name}.png"
        plt.savefig(fn, bbox_inches="tight")
        plt.close()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger.info("Running visualization inference on device: %s", device)
    
    # Ensure outputs directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. LOAD ALL CONFIGS AND INSTANTIATE MODELS
    models: dict[str, Any] = {}
    
    # Iteration 1 setup
    cfg1 = load_config(PROJECT_ROOT / "configs" / "iteration1.yaml")
    m1_cfg = cfg1.get("model", {"in_channels": 3, "num_conv_blocks": 4, "base_channels": 32})
    model1 = FireCNN(
        in_channels=m1_cfg.get("in_channels", 3),
        num_conv_blocks=m1_cfg.get("num_conv_blocks", 4),
        base_channels=m1_cfg.get("base_channels", 32),
    )
    chkpt1 = args.checkpoint_dir / "iteration1" / "best_model.pt"
    if load_model_weights(chkpt1, model1, device):
        models["iteration1"] = model1
        
    # Iteration 2 setup
    cfg2 = load_config(PROJECT_ROOT / "configs" / "iteration2.yaml")
    m2_cfg = cfg2.get("model", {"num_classes": 4, "dropout": 0.2})
    model2 = MobileNetV3FireClassifier(
        num_classes=m2_cfg.get("num_classes", 4),
        pretrained=False,
        dropout=m2_cfg.get("dropout", 0.2)
    )
    chkpt2 = args.checkpoint_dir / "iteration2" / "best_model.pt"
    if load_model_weights(chkpt2, model2, device):
        models["iteration2"] = model2

    # Iteration 3 setup
    cfg3 = load_config(PROJECT_ROOT / "configs" / "iteration3.yaml")
    m3_cfg = cfg3.get("model", {"num_classes": 4, "dropout": 0.2})
    model3 = MobileNetV3FireClassifier(
        num_classes=m3_cfg.get("num_classes", 4),
        pretrained=False,
        dropout=m3_cfg.get("dropout", 0.2)
    )
    chkpt3 = args.checkpoint_dir / "iteration3" / "best_model.pt"
    if load_model_weights(chkpt3, model3, device):
        models["iteration3"] = model3

    # Iteration 4 setup
    chkpt4 = args.checkpoint_dir / "iteration4" / "yolo26-fire" / "best.pt"
    model4 = load_iteration4_model(chkpt4)
    if model4 is not None:
        models["iteration4"] = model4

    # Iteration 5 setup
    cfg5 = load_config(PROJECT_ROOT / "configs" / "iteration5.yaml")
    m5_cfg = cfg5.get("model", {"in_channels": 3, "num_classes": 3, "base_channels": 32})
    model5 = LightweightUNet(
        in_channels=m5_cfg.get("in_channels", 3),
        num_classes=m5_cfg.get("num_classes", 3),
        base_channels=m5_cfg.get("base_channels", 32),
    )
    chkpt5 = args.checkpoint_dir / "iteration5" / "best_model.pt"
    if load_model_weights(chkpt5, model5, device):
        models["iteration5"] = model5
        
    if not models:
        logger.error("No models could be loaded! Please check if your checkpoints are saved in %s.", args.checkpoint_dir)
        sys.exit(1)
        
    # 2. DISCOVER / SELECT TEST IMAGES
    images_to_run: list[dict[str, Any]] = []
    
    if args.image_paths:
        for p in args.image_paths:
            if p.exists():
                images_to_run.append({
                    "path": p,
                    "ground_truth": "Custom",
                    "source": "CLI Argument"
                })
            else:
                logger.warning("Specified path does not exist: %s", p)
                
    elif args.input_dir and args.input_dir.is_dir():
        for p in args.input_dir.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                images_to_run.append({
                    "path": p,
                    "ground_truth": "Custom Folder",
                    "source": "CLI Input Folder"
                })
                
    elif not args.no_auto:
        # Auto-discover from project's data directory
        data_root = PROJECT_ROOT / "data"
        images_to_run = get_representative_images(data_root, args.num_samples)
        
    if not images_to_run:
        logger.error("No images found to process. Please provide images via --image-paths or --input-dir, "
                     "or check if the 'data/test' split directory exists for auto-discovery.")
        sys.exit(1)
        
    logger.info("Starting prediction runs on %d images...", len(images_to_run))
    
    # 3. RUN INFERENCE AND SAVE OUTPUTS
    for idx, sample in enumerate(images_to_run, start=1):
        img_path = sample["path"]
        ground_truth = sample["ground_truth"]
        source = sample["source"]
        
        logger.info("[%d/%d] Processing image %s (GT: %s, Source: %s)", idx, len(images_to_run), img_path.name, ground_truth, source)
        
        try:
            results = run_inference_on_image(img_path, models, device)
            save_visualizations(img_path.stem, results, ground_truth, args.output_dir)
        except Exception as e:
            logger.error("Failed to process image %s: %s", img_path.name, e, exc_info=True)
            
    logger.info("Visualization pipeline completed successfully!")
    logger.info("Results saved under: %s", args.output_dir.resolve())
    print(f"\nSuccess! Check out the generated images in your output folder:\n{args.output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
