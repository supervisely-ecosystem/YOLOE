from ultralytics.utils.downloads import attempt_download_asset

attempt_download_asset("mobileclip_blt.ts")
import supervisely as sly
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
