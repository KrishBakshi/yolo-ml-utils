import gradio as gr
from src.image_data_transformer.get_image_bbox_data_aug import process_bbox_aug
from src.image_data_transformer.get_image_seg_data_aug import process_seg_aug

with gr.Blocks(title="YOLO Data Augmentation Tools") as demo:
    gr.Markdown("## YOLO Data Augmentation Tools")

    with gr.Tabs():
        # Tab 1: Bounding box augmentation
        with gr.Tab("Bounding Box Augmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload & Settings")
                    with gr.Group():
                        with gr.Tab("ZIP File"):
                            bbox_zip_input = gr.File(
                                label="ZIP File (images + labels)",
                                file_types=[".zip"],
                                type="filepath",
                            )
                        with gr.Tab("Directory Path"):
                            bbox_path_input = gr.Textbox(
                                label="Directory Path",
                                placeholder="/path/to/directory/with/images/and/labels",
                                info="Path to directory containing 'images' and 'labels' folders",
                            )
                    bbox_angles_checkbox = gr.CheckboxGroup(
                        label="Rotation Angles",
                        choices=[90, 180, 270],
                        value=[90, 180, 270],
                        info="Select rotation angles to apply",
                    )
                    bbox_keep_original = gr.Checkbox(
                        value=True,
                        label="Keep Original Images",
                        info="Include original images in output",
                    )
                    bbox_submit_btn = gr.Button("Generate Augmentations", variant="primary")
                    bbox_text_output = gr.Textbox(
                        label="Status",
                        lines=8,
                        interactive=False,
                    )
                    bbox_download_btn = gr.File(
                        label="Download Augmented Data (ZIP)",
                        type="filepath",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Augmented Images")
                    bbox_image_gallery = gr.Gallery(
                        label="Augmented Images",
                        show_label=True,
                        elem_id="bbox_gallery",
                        columns=3,
                        rows=2,
                        height=600,
                        type="filepath",
                    )

                    with gr.Accordion("Instructions", open=False):
                        gr.Markdown(
                            "1. Upload a ZIP containing `images` and `labels` **OR** provide a directory path\n"
                            "2. Select rotation angles (90°, 180°, 270°)\n"
                            "3. Choose whether to keep original images\n"
                            "4. Click **Generate Augmentations**\n"
                            "5. View the gallery; download all as ZIP\n"
                            "6. Output ZIP contains augmented images and labels"
                        )

            def process_bbox_with_settings(zip_file, input_path, angles, keep_original):
                # Convert checkbox values to integers
                angle_list = [int(a) for a in angles] if angles else []
                return process_bbox_aug(zip_file, input_path, angle_list, keep_original)

            bbox_submit_btn.click(
                fn=process_bbox_with_settings,
                inputs=[bbox_zip_input, bbox_path_input, bbox_angles_checkbox, bbox_keep_original],
                outputs=[bbox_image_gallery, bbox_text_output, bbox_download_btn],
            )

        # Tab 2: Segmentation augmentation
        with gr.Tab("Segmentation Augmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload & Settings")
                    with gr.Group():
                        with gr.Tab("ZIP File"):
                            seg_zip_input = gr.File(
                                label="ZIP File (images + labels)",
                                file_types=[".zip"],
                                type="filepath",
                            )
                        with gr.Tab("Directory Path"):
                            seg_path_input = gr.Textbox(
                                label="Directory Path",
                                placeholder="/path/to/directory/with/images/and/labels",
                                info="Path to directory containing 'images' and 'labels' folders",
                            )
                    seg_angles_checkbox = gr.CheckboxGroup(
                        label="Rotation Angles",
                        choices=[90, 180, 270],
                        value=[90, 180, 270],
                        info="Select rotation angles to apply",
                    )
                    seg_keep_original = gr.Checkbox(
                        value=True,
                        label="Keep Original Images",
                        info="Include original images in output",
                    )
                    seg_submit_btn = gr.Button("Generate Augmentations", variant="primary")
                    seg_text_output = gr.Textbox(
                        label="Status",
                        lines=8,
                        interactive=False,
                    )
                    seg_download_btn = gr.File(
                        label="Download Augmented Data (ZIP)",
                        type="filepath",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Augmented Images")
                    seg_image_gallery = gr.Gallery(
                        label="Augmented Images",
                        show_label=True,
                        elem_id="seg_gallery",
                        columns=3,
                        rows=2,
                        height=600,
                        type="filepath",
                    )

                    with gr.Accordion("Instructions", open=False):
                        gr.Markdown(
                            "1. Upload a ZIP containing `images` and `labels` **OR** provide a directory path\n"
                            "2. Select rotation angles (90°, 180°, 270°)\n"
                            "3. Choose whether to keep original images\n"
                            "4. Click **Generate Augmentations**\n"
                            "5. View the gallery; download all as ZIP\n"
                            "6. Output ZIP contains augmented images and labels"
                        )

            def process_seg_with_settings(zip_file, input_path, angles, keep_original):
                # Convert checkbox values to integers
                angle_list = [int(a) for a in angles] if angles else []
                return process_seg_aug(zip_file, input_path, angle_list, keep_original)

            seg_submit_btn.click(
                fn=process_seg_with_settings,
                inputs=[seg_zip_input, seg_path_input, seg_angles_checkbox, seg_keep_original],
                outputs=[seg_image_gallery, seg_text_output, seg_download_btn],
            )

if __name__ == "__main__":
    demo.launch(share=False)
