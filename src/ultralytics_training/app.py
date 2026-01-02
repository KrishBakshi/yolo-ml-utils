import gradio as gr
import os
import shutil
import yaml
import random
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from ultralytics import YOLO

def create_train_val_split(data_path, val_ratio, output_dir, seed=42):
    """
    Create train/val split from a folder containing images and labels.
    Expected structure: data_path/images/*.png and data_path/labels/*.txt
    """
    try:
        data_path = Path(data_path)
        if not data_path.exists():
            return f"Error: Path {data_path} does not exist", None
        
        images_dir = data_path / "images"
        labels_dir = data_path / "labels"
        
        if not images_dir.exists():
            return f"Error: 'images' folder not found in {data_path}", None
        if not labels_dir.exists():
            return f"Error: 'labels' folder not found in {data_path}", None
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            return f"Error: No image files found in {images_dir}", None
        
        # Shuffle with seed
        random.seed(seed)
        image_files = sorted(image_files)
        random.shuffle(image_files)
        
        # Split
        n_total = len(image_files)
        n_val = max(1, int(round(val_ratio * n_total)))
        val_files = image_files[:n_val]
        train_files = image_files[n_val:]
        
        # Create output structure - default to parent directory of data_path if not specified
        if not output_dir or output_dir.strip() == "":
            # Default: create output in parent directory of data_path
            output_path = data_path.parent / f"{data_path.name}_split"
        else:
            output_path = Path(output_dir)
        
        train_images_dir = output_path / "train" / "images"
        train_labels_dir = output_path / "train" / "labels"
        val_images_dir = output_path / "val" / "images"
        val_labels_dir = output_path / "val" / "labels"
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy train files
        for img_file in train_files:
            shutil.copy2(img_file, train_images_dir / img_file.name)
            label_file = labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, train_labels_dir / label_file.name)
        
        # Copy val files
        for img_file in val_files:
            shutil.copy2(img_file, val_images_dir / img_file.name)
            label_file = labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, val_labels_dir / label_file.name)
        
        # Copy classes.txt if it exists
        classes_file = data_path / "classes.txt"
        if classes_file.exists():
            shutil.copy2(classes_file, output_path / "classes.txt")
            classes_info = "\n✓ Copied classes.txt to output folder"
        else:
            classes_info = "\n⚠ classes.txt not found in input folder"
        
        return f"Success! Created split:\nTrain: {len(train_files)} images\nVal: {len(val_files)} images{classes_info}\n\nSplit saved to: {output_path}", None
    
    except Exception as e:
        return f"Error: {str(e)}", None

def detect_yaml_files(split_folder_path):
    """Auto-detect YAML files in the split folder or parent directory."""
    try:
        if not split_folder_path:
            # If no path provided, search in common locations
            workspace_root = Path.cwd()
            yaml_files = list(workspace_root.glob("*.yaml")) + list(workspace_root.glob("*.yml"))
            return [str(f) for f in yaml_files]
        
        split_path = Path(split_folder_path)
        if not split_path.exists():
            # Try as file path
            if split_path.suffix in ['.yaml', '.yml']:
                if split_path.exists():
                    return [str(split_path)]
            return []
        
        # Priority order: dataset directory > parent directory > workspace root
        yaml_files_priority = []
        yaml_files_secondary = []
        
        # First priority: Check in split folder (dataset directory)
        if split_path.is_dir():
            dataset_yamls = list(split_path.glob("*.yaml")) + list(split_path.glob("*.yml"))
            yaml_files_priority.extend([f for f in dataset_yamls if f.exists()])
        
        # Second priority: Check in parent directory
        parent_path = split_path.parent
        parent_yamls = list(parent_path.glob("*.yaml")) + list(parent_path.glob("*.yml"))
        yaml_files_secondary.extend([f for f in parent_yamls if f.exists()])
        
        # Third priority: Check in project root (workspace root) - only if nothing found in dataset
        if not yaml_files_priority:
            workspace_root = Path.cwd()
            root_yamls = list(workspace_root.glob("*.yaml")) + list(workspace_root.glob("*.yml"))
            yaml_files_secondary.extend([f for f in root_yamls if f.exists()])
        
        # Combine: priority first, then secondary
        all_yamls = yaml_files_priority + yaml_files_secondary
        
        # Remove duplicates and return absolute paths
        unique_yamls = list(dict.fromkeys([str(f.resolve()) for f in all_yamls]))  # dict.fromkeys preserves order
        return unique_yamls if unique_yamls else []
    
    except Exception as e:
        return []

def read_classes_txt(folder_path):
    """Read class names from classes.txt file if it exists."""
    try:
        classes_file = Path(folder_path) / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
            return class_names
        return None
    except Exception as e:
        return None

def create_yaml_file(split_folder_path, class_names_str, yaml_filename):
    """Create a YAML file for the split folder."""
    try:
        split_path = Path(split_folder_path)
        if not split_path.exists():
            return f"Error: Path {split_folder_path} does not exist", None
        
        # Try to read from classes.txt first
        classes_from_file = read_classes_txt(split_path)
        
        # Parse class names from input
        class_names_input = [name.strip() for name in class_names_str.split(',') if name.strip()] if class_names_str else []
        
        # Use classes.txt if found, otherwise use input (input takes precedence if both provided)
        if classes_from_file and not class_names_input:
            class_names = classes_from_file
            source_info = f" (read from classes.txt)"
        elif class_names_input:
            class_names = class_names_input
            source_info = f" (from input)" + (f" - Note: classes.txt found but not used" if classes_from_file else "")
        else:
            return "Error: Please provide class names or ensure classes.txt exists in the folder", None
        
        if not class_names:
            return "Error: No class names found. Please provide class names or ensure classes.txt exists.", None
        
        # Get absolute path
        abs_path = split_path.resolve()
        
        # Create YAML content
        yaml_content = {
            'path': str(abs_path),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        # Save YAML file with custom formatting for names
        yaml_path = split_path / yaml_filename
        with open(yaml_path, 'w') as f:
            # Write YAML content manually to control format
            f.write(f"path: {abs_path}\n\n")
            f.write("# Train/validation split \n")
            f.write("train: train/images  \n")
            f.write("val: val/images    \n\n")
            f.write("# Number of classes\n")
            f.write(f"nc: {len(class_names)}\n\n")
            f.write("# Class names\n")
            # Format names as ['class1', 'class2', ...]
            names_str = ', '.join([f"'{name}'" for name in class_names])
            f.write(f"names: [{names_str}]\n")
        
        return f"Success! Created YAML file: {yaml_path}\nClasses: {', '.join(class_names)}{source_info}", str(yaml_path)
    
    except Exception as e:
        return f"Error: {str(e)}", None

def train_yolo_model(
    model_name,
    yaml_path,
    epochs,
    batch_size,
    imgsz,
    device,
    project_name,
    experiment_name,
    lr0,
    momentum,
    weight_decay,
    warmup_epochs,
    patience,
    workers
):
    """Train YOLO model with specified parameters."""
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    
    try:
        # Validate inputs
        if not model_name or not model_name.strip():
            return "Error: Model name is required", None
        
        # check if yaml_path is a file path or a yaml file
        if not yaml_path:
            return f"Error: YAML file not found: {yaml_path}", None
        
        # Start capturing output
        output_buffer.write(f"Starting training with model: {model_name}\n")
        output_buffer.write(f"YAML path: {yaml_path}\n")
        output_buffer.write(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}\n")
        output_buffer.write("-" * 50 + "\n")
        
        # Load model
        model = YOLO(model_name)
        
        # Prepare training arguments - handle empty strings as None
        train_args = {
            'data': yaml_path,
            'epochs': int(epochs),
            'batch': int(batch_size),
            'imgsz': int(imgsz),
        }
        
        # Add optional parameters only if provided
        if device and device.strip():
            train_args['device'] = device.strip()
        if project_name and project_name.strip():
            train_args['project'] = project_name.strip()
        if experiment_name and experiment_name.strip():
            train_args['name'] = experiment_name.strip()
        if lr0 and lr0.strip():
            train_args['lr0'] = float(lr0)
        if momentum and momentum.strip():
            train_args['momentum'] = float(momentum)
        if weight_decay and weight_decay.strip():
            train_args['weight_decay'] = float(weight_decay)
        if warmup_epochs and warmup_epochs.strip():
            train_args['warmup_epochs'] = float(warmup_epochs)
        if patience and patience.strip():
            train_args['patience'] = int(patience)
        if workers and workers.strip():
            train_args['workers'] = int(workers)
        
        # Capture stdout and stderr during training
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            # Train model
            results = model.train(**train_args)
        
        # Get captured output
        stdout_text = output_buffer.getvalue()
        stderr_text = error_buffer.getvalue()
        
        # Combine outputs
        full_output = stdout_text
        if stderr_text:
            full_output += "\n--- Errors/Warnings ---\n" + stderr_text
        
        # Get results path and add summary
        results_path = results.save_dir if hasattr(results, 'save_dir') else "runs"
        full_output += "\n" + "=" * 50 + "\n"
        full_output += f"Training completed successfully!\n"
        full_output += f"Results saved to: {results_path}\n"
        
        # Add results summary if available
        if hasattr(results, 'results_dict'):
            full_output += "\n--- Training Results Summary ---\n"
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    full_output += f"{key}: {value:.4f}\n"
        
        return full_output, results_path
    
    except Exception as e:
        error_msg = f"Error during training: {str(e)}\n"
        if output_buffer.getvalue():
            error_msg = output_buffer.getvalue() + "\n" + error_msg
        if error_buffer.getvalue():
            error_msg += "\n--- Error Details ---\n" + error_buffer.getvalue()
        return error_msg, None

# Gradio Interface
with gr.Blocks(title="YOLO Training Utility") as demo:
    gr.Markdown("# YOLO Training Utility")
    gr.Markdown("This app helps you create train/val splits and train YOLO models.")
    
    with gr.Tabs():
        
        # Tab 2: Create YAML File
        with gr.Tab("Create YAML File"):
            gr.Markdown("### Create YAML Configuration File")
            gr.Markdown("Create a YAML file for your train/val split folder.")
            
            with gr.Row():
                with gr.Column():
                    yaml_split_folder = gr.Textbox(
                        label="Split Folder Path (contains train/val)",
                        placeholder="/path/to/split/folder",
                        value="",
                        info="Enter folder path - classes.txt will be auto-detected if present"
                    )
                    with gr.Row():
                        yaml_class_names = gr.Textbox(
                            label="Class Names (comma-separated)",
                            placeholder="roi, room, text (or leave empty to use classes.txt)",
                            value="",
                            scale=4
                        )
                        yaml_detect_classes_btn = gr.Button("Auto-detect from classes.txt", size="sm", scale=1, variant="secondary")
                    yaml_filename = gr.Textbox(
                        label="YAML Filename",
                        value="dataset.yaml",
                        placeholder="dataset.yaml"
                    )
                    yaml_create_btn = gr.Button("Create YAML File", variant="primary")
                
                with gr.Column():
                    yaml_output = gr.Textbox(
                        label="Output",
                        lines=10,
                        interactive=False
                    )
                    yaml_file_output = gr.File(
                        label="Download YAML File",
                        visible=False
                    )
            
            def detect_classes_from_txt(folder_path):
                """Auto-detect and populate class names from classes.txt"""
                if not folder_path:
                    return "", "Please enter a folder path first"
                classes = read_classes_txt(folder_path)
                if classes:
                    class_names_str = ", ".join(classes)
                    return class_names_str, f"✓ Found {len(classes)} classes in classes.txt:\n" + "\n".join([f"  - {c}" for c in classes])
                return "", "✗ classes.txt not found in the specified folder"
            
            yaml_detect_classes_btn.click(
                fn=detect_classes_from_txt,
                inputs=[yaml_split_folder],
                outputs=[yaml_class_names, yaml_output]
            )
            
            yaml_create_btn.click(
                fn=create_yaml_file,
                inputs=[yaml_split_folder, yaml_class_names, yaml_filename],
                outputs=[yaml_output, yaml_file_output]
            ).then(
                fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False),
                inputs=[yaml_file_output],
                outputs=[yaml_file_output]
            )
        
        # Tab 3: Train YOLO Model
        with gr.Tab("Train YOLO Model"):
            gr.Markdown("### Train YOLO Model")
            gr.Markdown("Configure hyperparameters and train your YOLO model.")
            
            with gr.Row():
                with gr.Column():
                    # Model selection
                    model_name = gr.Textbox(
                        label="Model",
                        placeholder="example: yolo11n.pt",
                        info="Enter the model name, read the offcial docs at https://docs.ultralytics.com/models/ ",
                        value=""
                    )
                    
                    # YAML path with auto-detection
                    gr.Markdown("**YAML Configuration:**")
                    yaml_path_input = gr.Textbox(
                        label="YAML File Path",
                        placeholder="/path/to/dataset.yaml or /path/to/split/folder (for auto-detect)",
                        value="",
                        info="Enter YAML file path or folder path containing train/val splits"
                    )
                    with gr.Row():
                        yaml_auto_detect_btn = gr.Button("Auto-detect YAML", size="sm", variant="secondary")
                    yaml_detected = gr.Textbox(
                        label="Detected YAML Files",
                        lines=3,
                        interactive=False,
                        visible=True
                    )
                    
                    # Basic training parameters
                    
                
                with gr.Column():
                    with gr.Group():

                        with gr.Tab("Basic Training Parameters"):
                            epochs = gr.Number(label="Epochs", value=100, precision=0)
                            batch_size = gr.Number(label="Batch Size", value=16, precision=0)
                            imgsz = gr.Number(label="Image Size", value=640, precision=0)
                            device = gr.Textbox(
                                label="Device (cpu, 0, 1, etc.)",
                                placeholder="Leave empty for auto",
                                value=""
                            )
                            project_name = gr.Textbox(
                                label="Project Name",
                                placeholder="Leave empty for default",
                                value=""
                            )
                            experiment_name = gr.Textbox(
                                label="Experiment Name",
                                placeholder="Leave empty for default",
                                value=""
                            )
                        # Advanced hyperparameters
                        with gr.Tab("Advanced Hyperparameters"):
                            lr0 = gr.Textbox(
                                label="Initial Learning Rate (lr0)", 
                                placeholder="Default: 0.01", 
                                value="")

                            momentum = gr.Textbox(
                                label="Momentum", 
                                placeholder="Default: 0.937", 
                                value="")

                            weight_decay = gr.Textbox(
                                label="Weight Decay", 
                                placeholder="Default: 0.0005", 
                                value="")

                            warmup_epochs = gr.Textbox(
                                label="Warmup Epochs", 
                                placeholder="Default: 3.0", 
                                value="")

                            patience = gr.Textbox(
                                label="Early Stopping Patience", 
                                placeholder="Default: 100", 
                                value="")

                            workers = gr.Textbox(
                                label="Workers", 
                                placeholder="Default: 8", 
                                value="")
                            
                    train_btn = gr.Button("Start Training", variant="primary", size="lg")
                    
            train_output = gr.Textbox(
                label="Training Output",
                lines=30,
                interactive=False,
                max_lines=50
        )
            
            # Auto-detect YAML functionality
            def auto_detect_yaml(yaml_path_input_value):
                # Try to detect from the input (could be folder or file path)
                yaml_files = detect_yaml_files(yaml_path_input_value)
                if yaml_files:
                    yaml_list = "\n".join([f"- {f}" for f in yaml_files])
                    # If input is already a valid yaml file, use it; otherwise use first detected
                    if yaml_path_input_value and Path(yaml_path_input_value).exists() and Path(yaml_path_input_value).suffix in ['.yaml', '.yml']:
                        selected = yaml_path_input_value
                    else:
                        selected = yaml_files[0] if yaml_files else ""
                    return gr.update(value=selected, visible=True), yaml_list
                return gr.update(value=yaml_path_input_value, visible=True), "No YAML files found. Please create one in Tab 2 or provide a path."
            
            yaml_auto_detect_btn.click(
                fn=auto_detect_yaml,
                inputs=[yaml_path_input],
                outputs=[yaml_path_input, yaml_detected]
            ).then(
                fn=lambda x: gr.update(visible=True),
                inputs=[yaml_detected],
                outputs=[yaml_detected]
            )
            
            train_btn.click(
                fn=train_yolo_model,
                inputs=[
                    model_name,
                    yaml_path_input,
                    epochs,
                    batch_size,
                    imgsz,
                    device,
                    project_name,
                    experiment_name,
                    lr0,
                    momentum,
                    weight_decay,
                    warmup_epochs,
                    patience,
                    workers
                ],
                outputs=[train_output, gr.Textbox(visible=False)]
            )

if __name__ == "__main__":
    demo.launch(share=False)

