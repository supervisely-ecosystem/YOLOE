import supervisely as sly

sly.logger.info("Downloading mobileclip...")

import subprocess

subprocess.run(
    [
        "wget",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip_blt.ts",
    ],
    check=True,
)
from serve.src.yoloe_model import YOLOEModel
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv("supervisely.env")

model = YOLOEModel(
    use_gui=True,
    use_serving_gui_template=True,
)
model.serve()
