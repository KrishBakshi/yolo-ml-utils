import os
import tempfile
import zipfile
from pathlib import Path
import shutil
import random


def split_train_val_test(zip_file, input_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, 
                         seed=42, progress=None):
    """
    Split YOLO dataset (ZIP or directory) into train/val/test sets
    
    Args:
        zip_file: Path to ZIP file (or None if using directory path)
        input_path: Path to directory (or None if using ZIP file)
        train_ratio: Ratio for training set (0 to skip)
        val_ratio: Ratio for validation set (0 to skip)
        test_ratio: Ratio for test set (0 to skip)
        seed: Random seed for reproducibility (default: 42)
        progress: Gradio progress tracker (optional)
    
    Returns:
        tuple: (output_dir, status_msg, zip_path) or (None, error_msg, None) on error
    """
    if zip_file is None and (input_path is None or input_path.strip() == ""):
        return None, "Error: Please provide either a ZIP file or a directory path", None
    
    # Filter out zero ratios
    ratios = []
    split_names = []
    if train_ratio > 0:
        ratios.append(train_ratio)
        split_names.append('train')
    if val_ratio > 0:
        ratios.append(val_ratio)
        split_names.append('val')
    if test_ratio > 0:
        ratios.append(test_ratio)
        split_names.append('test')
    
    if not ratios:
        return None, "Error: At least one split ratio must be greater than 0", None
    
    # Validate ratios sum to 1.0
    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 0.001:
        return None, f"Error: Non-zero ratios must sum to 1.0 (got {total_ratio})", None
    
    output_dir = tempfile.mkdtemp(prefix="yolo_split_")
    extract_dir = None
    is_zip = zip_file is not None
    
    try:
        # Determine source directory
        if is_zip:
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            source_dir = extract_dir
        else:
            input_path = input_path.strip()
            if not os.path.isdir(input_path):
                shutil.rmtree(output_dir)
                return None, f"Error: Invalid directory path: {input_path}", None
            source_dir = input_path
        
        # Validate YOLO structure
        images_dir = os.path.join(source_dir, "images")
        labels_dir = os.path.join(source_dir, "labels")
        
        if not os.path.isdir(images_dir):
            if extract_dir:
                shutil.rmtree(extract_dir)
            shutil.rmtree(output_dir)
            return None, "Error: 'images' directory not found", None
        
        if not os.path.isdir(labels_dir):
            if extract_dir:
                shutil.rmtree(extract_dir)
            shutil.rmtree(output_dir)
            return None, "Error: 'labels' directory not found", None
        
        # Get image files count
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_files = [f for f in Path(images_dir).iterdir() 
                      if f.suffix in image_extensions]
        
        if not image_files:
            if extract_dir:
                shutil.rmtree(extract_dir)
            shutil.rmtree(output_dir)
            return None, "Error: No image files found in images directory", None
        
        # Manually split files to ensure image-label pairs stay together
        # Convert Path objects to strings for easier handling
        image_files_list = [str(img) for img in image_files]
        
        # Set random seed for reproducibility
        random.seed(seed)
        random.shuffle(image_files_list)
        
        # Calculate split indices
        total = len(image_files_list)
        splits = {}
        start_idx = 0
        
        for i, split_name in enumerate(split_names):
            ratio = ratios[i]
            count = int(total * ratio)
            if i == len(split_names) - 1:
                # Last split gets remaining files to account for rounding
                count = total - start_idx
            splits[split_name] = image_files_list[start_idx:start_idx + count]
            start_idx += count
        
        # Create output directories and copy files
        for split_name in split_names:
            images_out = os.path.join(output_dir, split_name, "images")
            labels_out = os.path.join(output_dir, split_name, "labels")
            os.makedirs(images_out, exist_ok=True)
            os.makedirs(labels_out, exist_ok=True)
            
            # Copy images and corresponding labels
            for img_path in splits[split_name]:
                img = Path(img_path)
                img_name = img.name
                img_stem = img.stem
                
                # Copy image
                shutil.copy2(img_path, os.path.join(images_out, img_name))
                
                # Copy corresponding label if exists
                label_file = os.path.join(labels_dir, f"{img_stem}.txt")
                if os.path.exists(label_file):
                    shutil.copy2(label_file, os.path.join(labels_out, f"{img_stem}.txt"))
        
        # Copy classes.txt if exists
        classes_file = os.path.join(source_dir, "classes.txt")
        if os.path.exists(classes_file):
            shutil.copy2(classes_file, os.path.join(output_dir, "classes.txt"))
        
        # Cleanup
        if extract_dir:
            shutil.rmtree(extract_dir)
        
        # Count files in each split
        split_counts = {}
        for split_name in split_names:
            img_dir = os.path.join(output_dir, split_name, "images")
            if os.path.exists(img_dir):
                split_counts[split_name] = len(list(Path(img_dir).iterdir()))
            else:
                split_counts[split_name] = 0
        
        # Create ZIP archive
        zip_path = os.path.join(output_dir, "yolo_dataset_split.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file == "yolo_dataset_split.zip":
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        
        # Generate status message
        status_msg = "Successfully split YOLO dataset!\n\n"
        status_msg += f"Input: {Path(zip_file).name if is_zip else input_path}\n"
        status_msg += f"\nSplit distribution:\n"
        for split_name in split_names:
            ratio = train_ratio if split_name == 'train' else (val_ratio if split_name == 'val' else test_ratio)
            count = split_counts.get(split_name, 0)
            status_msg += f"  {split_name.capitalize()}: {ratio*100:.1f}% ({count} images)\n"
        status_msg += f"\nTotal images: {len(image_files)}\n"
        status_msg += f"Random seed: {seed}\n"
        
        return output_dir, status_msg, zip_path
        
    except zipfile.BadZipFile:
        if extract_dir:
            shutil.rmtree(extract_dir)
        shutil.rmtree(output_dir)
        return None, "Error: Invalid ZIP file format", None
    except Exception as e:
        if extract_dir:
            shutil.rmtree(extract_dir)
        shutil.rmtree(output_dir)
        return None, f"Error: {str(e)}", None