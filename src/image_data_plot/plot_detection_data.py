import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def get_class_color(class_id):
    """Get consistent color for a class"""
    if class_id < len(CUSTOM_COLORS):
        return CUSTOM_COLORS[class_id]
    # If more classes than palette, cycle through colors
    return CUSTOM_COLORS[class_id % len(CUSTOM_COLORS)]

def plot_yolo(image_path, label_path, output_path, class_names=None):
    """Plot YOLO annotations"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    
    # Resize image if too large to prevent memory issues
    max_size = 1024
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_rgb = cv2.resize(img_rgb, (new_width, new_height))
        height, width = new_height, new_width
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    # Read labels
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Plot annotations
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_color = get_class_color(class_id)  # Use consistent colors
                    # Convert BGR (0-255) to RGB (0-1) for matplotlib
                    color_rgb = [class_color[2] / 255.0, class_color[1] / 255.0, class_color[0] / 255.0]
                    
                    if len(parts) == 5:
                        # Bounding box
                        x_center, y_center, w, h = map(float, parts[1:5])
                        x1 = (x_center - w/2) * width
                        y1 = (y_center - h/2) * height
                        w_px = w * width
                        h_px = h * height
                        
                        rect = patches.Rectangle((x1, y1), w_px, h_px, 
                                               linewidth=0.8, 
                                               edgecolor=color_rgb, 
                                               facecolor='none')
                        ax.add_patch(rect)
                        
                        # Label
                        label = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
                        ax.text(x1, y1-5, label, fontsize=8, color=color_rgb,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    else:
                        # Segmentation
                        points = []
                        for i in range(1, len(parts), 2):
                            if i + 1 < len(parts):
                                x = float(parts[i]) * width
                                y = float(parts[i + 1]) * height
                                points.append((x, y))
                        
                        if len(points) > 2:
                            print("The given dataset appears to be a segmentation dataset, not a detection dataset.")
                            print("Please use the Segmentation Plotter instead.")
                            # polygon = patches.Polygon(points, 
                            #                         linewidth=0.8, 
                            #                         edgecolor=color_rgb, 
                            #                         facecolor=color_rgb, 
                            #                         alpha=0.3)
                            # ax.add_patch(polygon)
                            
                            # # Label
                            # center_x = sum(p[0] for p in points) / len(points)
                            # center_y = sum(p[1] for p in points) / len(points)
                            # label = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
                            # ax.text(center_x, center_y, label, fontsize=8, color='white',
                            #        bbox=dict(boxstyle="round,pad=0.2", facecolor=color_rgb, alpha=0.8))
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    ax.set_title(f"YOLO Annotations: {Path(image_path).name}", fontsize=12)
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def process_detection_zip(zip_file, input_path, progress=gr.Progress(track_tqdm=True)):
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
        progress_lines = [f"Processing {len(image_files)} detection image(s)..."]
        pbar = progress.tqdm(image_files, desc="Processing detection images")
        for idx, img_path in enumerate(pbar, 1):
            img_name = img_path.stem
            label_path = os.path.join(labels_dir, f"{img_name}.txt")
            output_path = os.path.join(temp_dir, f"{img_name}_plot.png")
            
            try:
                plot_yolo(str(img_path), label_path, output_path, class_names)
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
        zip_path = os.path.join(temp_dir, "yolo_plots.zip")
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
        status_msg += f"Processed images: {', '.join(processed_images)}\n"
        if class_names:
            status_msg += f"Classes: {', '.join(class_names)}\n"
        
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