import gradio as gr
import gradio as gr
import yaml
import importlib

def load_apps_from_config(config_path="config/app_registry.yaml"):
    with open(config_path, "r") as f:
        apps_config = yaml.safe_load(f)
    
    apps = []
    for entry in apps_config:
        app_data = {
            "name": entry["name"],
            "id": entry["id"],
            "render_fn": None
        }
        
        if entry.get("import_path"):
            try:
                module = importlib.import_module(entry["import_path"])
                # Assumes the module has a 'app' object with a 'demo' block
                # Adjust based on actual structure. 
                # The previous code used: src.image_data_plot.app.demo.render
                # Check if module has 'demo' attribute directly or via 'app'
                if hasattr(module, "demo"):
                     app_data["render_fn"] = module.demo.render
                elif hasattr(module, "app") and hasattr(module.app, "demo"):
                     app_data["render_fn"] = module.app.demo.render
            except ImportError as e:
                print(f"Error loading module {entry['import_path']}: {e}")
                
        apps.append(app_data)
    return apps

APPS = load_apps_from_config()

# Defines the registry of apps

def render_home():
    """Renders the static home content."""
    gr.Markdown("# ML Utils")
    gr.Markdown("*Machine Learning Utilities and Visualization Tools*")
    gr.Markdown("---")
    gr.Markdown("### Welcome!")
    gr.Markdown("Use the sidebar to navigate to different tools.")
    gr.Markdown("### Available Tools:")
    for app in APPS:
        if app["id"] != "home":
             gr.Markdown(f"- **{app['name']}**")

with gr.Blocks(title="ML Utils") as demo:
    # Store buttons to wire up click events later
    nav_buttons = []

    # Sidebar
    with gr.Sidebar():
        gr.Markdown("## ML Utils")
        gr.Markdown("---")
        gr.Markdown("### Navigation")
        
        for app in APPS:
            btn = gr.Button(app['name'], variant="secondary")
            nav_buttons.append((btn, app["id"]))
            
        gr.Markdown("---")

    # Container to hold all app views
    app_views = []

    # Main content
    # We use Columns for each app and toggle visibility
    for app in APPS:
        with gr.Column(visible=(app["id"] == "home")) as view:
            if app["id"] == "home":
                render_home()
            elif app["render_fn"]:
                app["render_fn"]()
        app_views.append(view)

    # Navigation wiring
    def switch_view(target_id):
        """Returns a list of updates for all app views."""
        updates = []
        for app in APPS:
            # Show the view if it matches targetId, hide otherwise
            updates.append(gr.update(visible=(app["id"] == target_id)))
        return updates

    for btn, app_id in nav_buttons:
        # Click updates ALL views' visibility
        btn.click(
            fn=lambda t=app_id: switch_view(t),
            outputs=app_views,
            show_progress=False
        )

if __name__ == "__main__":
    demo.launch(share=False)

