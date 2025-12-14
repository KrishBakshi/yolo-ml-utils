import os
import cv2
from pathlib import Path
import tempfile
import zipfile
import shutil
import gradio as gr
from src.image_data_transformer.get_image_bbox_data_aug import copy_original_image_and_labels

#Aug for segmatation data
def rotate_segmentation_labels(segmentation_points, angle, img_width, img_height):
    """Rotate segmentation polygon points"""
    if angle == 90:
        # 90 degrees clockwise: (x,y) -> (y, width-x)
        return [(img_width - y, x) for x, y in segmentation_points]
    elif angle == 180:
        # 180 degrees: (x,y) -> (width-x, height-y)
        return [(img_width - x, img_height - y) for x, y in segmentation_points]
    elif angle == 270:
        # 270 degrees clockwise (90 counter-clockwise): (x,y) -> (y, height-x)
        return [(y,img_height -  x) for x, y in segmentation_points]
    return segmentation_points

def parse_yolo_segmentation(line):
    """Parse YOLO segmentation line"""
    parts = line.strip().split()
    class_id = int(parts[0])
    points = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            x = float(parts[i])
            y = float(parts[i + 1])
            points.append((x, y))
    return class_id, points

def format_yolo_segmentation(class_id, points):
    """Format segmentation points back to YOLO format"""
    result = [str(class_id)]
    for x, y in points:
        result.extend([f"{x:.6f}", f"{y:.6f}"])
    return " ".join(result)

def rotate_image_and_segmentation(image_path, label_path, output_dir, angle):
    """Rotate image and its segmentation labels"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    height, width = img.shape[:2]
    
    # Rotate image
    if angle == 90:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        new_width, new_height = height, width
    elif angle == 180:
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        new_width, new_height = width, height
    elif angle == 270:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_width, new_height = height, width
    else:
        return
    
    # Process labels
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        rotated_labels = []
        for line in lines:
            if line.strip():
                class_id, points = parse_yolo_segmentation(line)
                
                # Convert normalized coordinates to pixel coordinates
                pixel_points = [(x * width, y * height) for x, y in points]
                
                # Rotate points
                rotated_pixel_points = rotate_segmentation_labels(pixel_points, angle, width, height)
                
                # Convert back to normalized coordinates for the NEW image dimensions
                normalized_points = [(x / new_width, y / new_height) for x, y in rotated_pixel_points]
                
                # Format and add to labels
                rotated_labels.append(format_yolo_segmentation(class_id, normalized_points))
    
    # Save rotated image
    base_name = Path(image_path).stem
    output_img_name = f"{base_name}_rot{angle}.png"
    output_img_path = os.path.join(output_dir, "images", output_img_name)
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    cv2.imwrite(output_img_path, rotated_img)
    
    # Save rotated labels
    if os.path.exists(label_path):
        output_label_name = f"{base_name}_rot{angle}.txt"
        output_label_path = os.path.join(output_dir, "labels", output_label_name)
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(rotated_labels))

def process_seg_aug(zip_file, input_path, angles, keep_original, progress=gr.Progress(track_tqdm=True)):
    """Process YOLO data (ZIP or path) for segmentation augmentation and return results"""
    if zip_file is None and (input_path is None or input_path.strip() == ""):
        return [], "Error: Please provide either a ZIP file or a directory path", None
    
    if not angles:
        return [], "Error: Please select at least one rotation angle", None
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()  # For output images
    extract_dir = None
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
        progress_lines = [f"Processing {len(image_files)} image(s) with angles: {', '.join(map(str, angles))}..."]
        pbar = progress.tqdm(image_files, desc="Processing segmentation augmentation")
        
        for idx, img_path in enumerate(pbar, 1):
            img_name = img_path.stem
            label_path = os.path.join(labels_dir, f"{img_name}.txt")
            
            try:
                # Copy original if requested
                if keep_original:
                    copy_original_image_and_labels(str(img_path), label_path, temp_dir)
                    output_paths.append(os.path.join(temp_dir, "images", f"{img_name}.png"))
                
                # Apply rotations
                for angle in angles:
                    rotate_image_and_segmentation(str(img_path), label_path, temp_dir, angle)
                    output_paths.append(os.path.join(temp_dir, "images", f"{img_name}_rot{angle}.png"))
                
                processed_images.append(img_name)
                progress_lines.append(
                    f"[{idx}/{len(image_files)}] {img_name}"
                )
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
        
        if not output_paths:
            if extract_dir:
                shutil.rmtree(extract_dir)
            shutil.rmtree(temp_dir)
            return [], f"Error: Failed to process any images", None
        
        # Create ZIP file with all augmented images and labels
        zip_path = os.path.join(temp_dir, "seg_augmented.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add images
            images_output_dir = os.path.join(temp_dir, "images")
            if os.path.exists(images_output_dir):
                for img_file in Path(images_output_dir).glob("*"):
                    zipf.write(img_file, os.path.join("images", img_file.name))
            
            # Add labels
            labels_output_dir = os.path.join(temp_dir, "labels")
            if os.path.exists(labels_output_dir):
                for label_file in Path(labels_output_dir).glob("*"):
                    zipf.write(label_file, os.path.join("labels", label_file.name))
            
            # Copy classes.txt if it exists
            if is_zip:
                classes_file = os.path.join(extract_dir, "classes.txt")
            else:
                classes_file = os.path.join(input_path, "classes.txt")
            if os.path.exists(classes_file):
                zipf.write(classes_file, "classes.txt")
        
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
        status_msg += f"Rotation angles: {', '.join(map(str, angles))}\n"
        status_msg += f"Keep original: {keep_original}\n"
        status_msg += f"Total augmented images: {len(output_paths)}\n"
        
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