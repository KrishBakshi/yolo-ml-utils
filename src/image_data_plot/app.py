import gradio as gr
from src.image_data_plot.plot_detection_data import process_detection_zip as process_detection_zip
from src.image_data_plot.plot_segmentation_data import process_segmentation_zip as process_segmentation_zip

with gr.Blocks(title="YOLO Visualization Tools") as demo:
    gr.Markdown("## YOLO Visualization Tools")

    with gr.Tabs():
        # Tab 1: Detection plots (matplotlib)
        with gr.Tab("Bounding Box Visualizer"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload & Run")
                    with gr.Group():
                        with gr.Tab("ZIP File"):
                            mpl_zip_input = gr.File(
                                label="ZIP File (images + labels)",
                                file_types=[".zip"],
                                type="filepath",
                            )
                        with gr.Tab("Directory Path"):
                            mpl_path_input = gr.Textbox(
                                label="Directory Path",
                                placeholder="/path/to/directory/with/images/and/labels",
                                info="Path to directory containing 'images' and 'labels' folders",
                            )
                    mpl_submit_btn = gr.Button("Generate Plots", variant="primary")
                    mpl_text_output = gr.Textbox(
                        label="Status",
                        lines=8,
                        interactive=False,
                    )
                    mpl_download_btn = gr.File(
                        label="Download All Plots (ZIP)",
                        type="filepath",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Visualizations")
                    mpl_image_gallery = gr.Gallery(
                        label="Plotted Images",
                        show_label=True,
                        elem_id="mpl_gallery",
                        columns=3,
                        rows=2,
                        height=600,
                        type="filepath",
                    )

                    with gr.Accordion("Instructions", open=False):
                        gr.Markdown(
                            "1. Upload a ZIP containing `images` and `labels`\n"
                            "2. Click **Generate Plots**\n"
                            "3. View the gallery; download all as ZIP\n"
                            "4. `classes.txt` (optional) is used for labels"
                        )

            mpl_submit_btn.click(
                fn=process_detection_zip,
                inputs=[mpl_zip_input, mpl_path_input],
                outputs=[mpl_image_gallery, mpl_text_output, mpl_download_btn],
            )

        # Tab 2: Segmentation plots (OpenCV)
        with gr.Tab("Segmentation Visualizer"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload & Settings")
                    with gr.Group():
                        with gr.Tab("ZIP File"):
                            cv_zip_input = gr.File(
                                        label="ZIP File (images + labels)",
                                        file_types=[".zip"],
                                        type="filepath",
                            )
                        with gr.Tab("Directory Path"):
                            cv_path_input = gr.Textbox(
                                label="Directory Path",
                                placeholder="/path/to/directory/with/images/and/labels",
                                info="Path to directory containing 'images' and 'labels' folders",
                            )
                    segmentation_type_dropdown = gr.Dropdown(
                        label="Segmentation Type",
                        choices=["Semantic", "Instance", "Panoptic"],
                        value="Semantic",
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Transparency (Alpha)",
                        info="Lower values make overlays more transparent",
                    )
                    show_polygons_checkbox = gr.Checkbox(
                        value=True,
                        label="Show Polygon Outlines",
                    )
                    cv_submit_btn = gr.Button("Generate Visualizations", variant="primary")
                    cv_text_output = gr.Textbox(
                        label="Status",
                        lines=8,
                        interactive=False,
                    )
                    cv_download_btn = gr.File(
                        label="Download All Plots (ZIP)",
                        type="filepath",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Visualizations")
                    cv_image_gallery = gr.Gallery(
                        label="Segmentation Visualizations",
                        show_label=True,
                        elem_id="cv_gallery",
                        columns=3,
                        rows=2,
                        height=600,
                        type="filepath",
                    )

                    with gr.Accordion("Instructions", open=False):
                        gr.Markdown(
                            "1. Upload a ZIP containing `images` and `labels`\n"
                            "2. Adjust transparency and polygon outline settings\n"
                            "3. Click **Generate Visualizations**\n"
                            "4. View the gallery; download all as ZIP\n"
                            "5. `classes.txt` (optional) is used for labels"
                        )

            def process_with_settings_zip(zip_file, input_path, alpha, show_polygons, segmentation_type):
                return process_segmentation_zip(zip_file, input_path, alpha=alpha, show_polygons=show_polygons, segmentation_type=segmentation_type)

            cv_submit_btn.click(
                fn=process_with_settings_zip,
                inputs=[cv_zip_input, cv_path_input, alpha_slider, show_polygons_checkbox, segmentation_type_dropdown],
                outputs=[cv_image_gallery, cv_text_output, cv_download_btn],
            )

if __name__ == "__main__":
    demo.launch(share=False)