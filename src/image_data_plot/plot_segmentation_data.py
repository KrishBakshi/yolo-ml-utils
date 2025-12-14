import os
import cv2
import numpy as np
import tempfile
import zipfile
from pathlib import Path
import shutil
import yaml
import gradio as gr
from pathlib import Path
from tqdm.auto import tqdm  # tqdm still used, but tracked by gradio

# Load custom colors with proper path resolution
config_path = Path(__file__).parent.parent.parent / "config" / "cycle_colors.yaml"
with open(config_path, 'r') as f:
    CUSTOM_COLORS = yaml.safe_load(f)['CUSTOM_COLORS']

def create_segmentation_visualization(
    image_path: str,
    label_path: str,
    output_path: str,
    alpha: float = 0.3,
    show_polygons: bool = True,
    class_names: list = None,
    segmentation_type: str = "Semantic"
) -> str:
    """
    Create segmentation visualization from YOLO format labels using OpenCV
    
    Args:
        image_path: Path to input image
        label_path: Path to YOLO format label file
        output_path: Path to save visualization
        alpha: Transparency level (0.0 = fully transparent, 1.0 = fully opaque)
        show_polygons: Whether to draw polygon outlines
        class_names: List of class names (optional)
        segmentation_type: Type of segmentation - "Semantic", "Instance", or "Panoptic"
    
    Returns:
        Path to saved visualization
    """
    # Read image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_img_rgb.shape[:2]
    
    # Start with original image as background
    colored_mask = original_img_rgb.copy().astype(np.float32)
    
    # Read labels
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Process each annotation
        for seg_idx, line in enumerate(lines):
            if not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class_id + 6 coordinates (3 points minimum)
                continue
            
            class_id = int(parts[0])
            
            # Parse polygon coordinates (normalized)
            coords = [float(x) for x in parts[1:]]
            
            # Convert normalized coordinates to pixel coordinates
            pixel_coords = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * orig_w)
                    y = int(coords[i + 1] * orig_h)
                    pixel_coords.append([x, y])
            
            if len(pixel_coords) < 3:
                continue
            
            # Create mask from polygon
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            pts = np.array(pixel_coords, np.int32)
            cv2.fillPoly(mask, [pts], 255)
            
            # Get color for this segment based on segmentation type
            if segmentation_type == "Semantic":
                # Semantic: All polygons of the same class use the same color
                color_idx = class_id % len(CUSTOM_COLORS)
            elif segmentation_type == "Instance":
                # Instance: Each polygon gets a unique color (by segment index)
                color_idx = seg_idx % len(CUSTOM_COLORS)
            elif segmentation_type == "Panoptic":
                # Panoptic: Combine class_id and seg_idx for unique colors per instance
                # Use class_id for base color, seg_idx for variation
                color_idx = (class_id * 10 + seg_idx) % len(CUSTOM_COLORS)
            else:
                # Default to semantic if unknown type
                color_idx = class_id % len(CUSTOM_COLORS)
            
            color = np.array(CUSTOM_COLORS[color_idx], dtype=np.float32)
            
            # Normalize mask to 0-1 range
            mask_normalized = mask / 255.0
            
            # Apply mask with color and transparency
            for c in range(3):
                colored_mask[:, :, c] = np.where(
                    mask_normalized > 0.5,
                    alpha * color[c] + (1 - alpha) * colored_mask[:, :, c],
                    colored_mask[:, :, c]
                )
            
            # Draw polygon outlines if requested
            if show_polygons:
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw polygon for each contour
                for contour in contours:
                    # Simplify contour to reduce points
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Convert to uint8 for drawing
                    colored_mask_uint8 = np.clip(colored_mask, 0, 255).astype(np.uint8)
                    color_bgr = [int(color[2]), int(color[1]), int(color[0])]  # Convert RGB to BGR for OpenCV
                    cv2.polylines(colored_mask_uint8, [approx_polygon], isClosed=True, 
                                 color=color_bgr, thickness=2)
                    colored_mask = colored_mask_uint8.astype(np.float32)
    
    # Convert back to uint8 and BGR for saving
    vis_img = np.clip(colored_mask, 0, 255).astype(np.uint8)
    vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    # Save visualization
    cv2.imwrite(output_path, vis_img_bgr)
    
    return output_path


def process_segmentation_zip(zip_file, input_path, alpha=0.3, show_polygons=True, segmentation_type="Semantic", progress=gr.Progress(track_tqdm=True)):
    """Process YOLO zip file and return results"""
    if zip_file is None and (input_path is None or input_path.strip() == ""):
        return [], "Error: Please provide either a ZIP file or a directory path", None
    
    # Create temporary directory for extraction
    extract_dir = None
    temp_dir = tempfile.mkdtemp()  # For output plots
    is_zip = zip_file is not None

    try:
        if is_zip:
            # Handle ZIP file
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            images_dir = os.path.join(extract_dir, "images")
            labels_dir = os.path.join(extract_dir, "labels")
        else:
            # Handle directory path
            input_path = input_path.strip()
            if not os.path.exists(input_path):
                shutil.rmtree(temp_dir)
                return [], f"Error: Path does not exist: {input_path}", None
            if not os.path.isdir(input_path):
                shutil.rmtree(temp_dir)
                return [], f"Error: Path is not a directory: {input_path}", None
            images_dir = os.path.join(input_path, "images")
            labels_dir = os.path.join(input_path, "labels")
        
        if not os.path.exists(images_dir):
            if extract_dir:
                shutil.rmtree(extract_dir)
            shutil.rmtree(temp_dir)
            input_type = "ZIP file" if is_zip else "directory"
            return [], f"Error: 'images' directory not found in {input_type}", None
        if not os.path.exists(labels_dir):
            if extract_dir:
                shutil.rmtree(extract_dir)
            shutil.rmtree(temp_dir)
            input_type = "ZIP file" if is_zip else "directory"
            return [], f"Error: 'labels' directory not found in {input_type}", None
        
        # Look for classes.txt in the extracted directory
        if is_zip:
            classes_file = os.path.join(extract_dir, "classes.txt")
        else:
            classes_file = os.path.join(input_path, "classes.txt")

        class_names = None
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        
        # Get image files
        image_files = list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg")) + \
                     list(Path(images_dir).glob("*.jpeg")) + list(Path(images_dir).glob("*.JPG"))
        
        if not image_files:
            if extract_dir:
                shutil.rmtree(extract_dir)
            shutil.rmtree(temp_dir)
            return [], f"Error: No image files found in images directory", None
        
        output_paths = []
        processed_images = []
        
        progress_lines = [f"Processing {len(image_files)} segmentation image(s) ({segmentation_type} mode)..."]
        pbar = progress.tqdm(image_files, desc="Processing segmentation images")
        for idx, img_path in enumerate(pbar, 1):
            img_name = img_path.stem
            label_path = os.path.join(labels_dir, f"{img_name}.txt")
            output_path = os.path.join(temp_dir, f"{img_name}_segmentation.png")
            
            try:
                create_segmentation_visualization(
                    str(img_path), 
                    label_path, 
                    output_path, 
                    alpha=alpha,
                    show_polygons=show_polygons,
                    class_names=class_names,
                    segmentation_type=segmentation_type
                )
                output_paths.append(output_path)
                processed_images.append(img_name)
                progress_lines.append(
                    f"[{idx}/{len(image_files)}] {img_name}"
                )
            except Exception as e:
                print(f"Error plotting {img_name}: {e}")
                continue
        
        if not output_paths:
            if extract_dir:
                shutil.rmtree(extract_dir)
            shutil.rmtree(temp_dir)
            return [], f"Error: Failed to process any images", None
        
        # Create ZIP file with all plots
        zip_path = os.path.join(temp_dir, f"{segmentation_type}_segmentation_plots.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for plot_path in output_paths:
                zipf.write(plot_path, os.path.basename(plot_path))
        
        # Clean up extracted directory if it was a ZIP
        if extract_dir:
            shutil.rmtree(extract_dir)
        
        # Generate status message
        status_msg = "\n".join(progress_lines) + "\n\n"
        status_msg += f"Successfully processed {'ZIP file' if is_zip else 'directory'}!\n\n"
        if is_zip:
            status_msg += f"ZIP file: {Path(zip_file).name}\n"
        else:
            status_msg += f"Directory: {input_path}\n"
        status_msg += f"Found {len(image_files)} image(s)\n"
        status_msg += f"Processed {len(processed_images)} image(s) successfully\n"
        status_msg += f"Processed images: {', '.join(processed_images[:10])}"  # Show first 10
        if len(processed_images) > 10:
            status_msg += f" ... and {len(processed_images) - 10} more\n"
        if class_names:
            status_msg += f"\nClasses: {', '.join(class_names)}\n"
        
        # Return all images for gallery and ZIP file for download
        return output_paths, status_msg, zip_path
        
    except zipfile.BadZipFile:
        if extract_dir and os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return [], "Error: Invalid ZIP file format", None
    except Exception as e:
        if extract_dir and os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        input_type = "ZIP file" if is_zip else "directory"
        return [], f"Error processing {input_type}: {str(e)}", None