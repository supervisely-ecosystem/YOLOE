import supervisely as sly
from ultralytics import YOLOE
import os
from supervisely.nn.inference import CheckpointInfo, ModelSource, RuntimeType
import torch
from supervisely.nn.inference.inference import get_hardware_info
import numpy as np
from typing import List
import cv2
from supervisely.nn.prediction_dto import PredictionBBox
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


class YOLOEModel(sly.nn.inference.PromptBasedObjectDetection):
    FRAMEWORK_NAME = "YOLOE"
    MODELS = "models/models.json"
    APP_OPTIONS = "serve/src/app_options.yaml"
    INFERENCE_SETTINGS = "serve/src/inference_settings.yaml"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # disable GUI widgets
        self.gui.set_project_meta = self.set_project_meta

    def load_model(
        self,
        model_files: dict,
        model_info: dict,
        model_source: str,
        device: str,
        runtime: str,
    ):
        self.checkpoint_path = model_files["weights_url"]
        self.device = device
        self.model = YOLOE(self.checkpoint_path)
        self.model.to(self.device)
        self.model.model.is_fused = lambda: True
        self.model_type = model_info["prompt"]

        if model_source == ModelSource.PRETRAINED:
            self.checkpoint_info = CheckpointInfo(
                checkpoint_name=os.path.basename(self.checkpoint_path),
                model_name=model_info["meta"]["model_name"],
                architecture=self.FRAMEWORK_NAME,
                checkpoint_url=model_info["meta"]["model_files"]["weights_url"],
                model_source=model_source,
            )

        if self.model_type == "text/visual":
            self.classes = ["object"]
        else:
            self.classes = list(self.model.names.values())
        obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in self.classes]
        conf_tag = sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
        self._model_meta = sly.ProjectMeta(
            obj_classes=obj_classes, tag_metas=[conf_tag]
        )
        self.reference_image_id = None
        self.reference_image = None

    def predict_benchmark(self, images_np: List[np.ndarray], settings: dict = None):
        if self.runtime == RuntimeType.PYTORCH:
            return self._predict_pytorch(images_np, settings)

    def _predict_pytorch(self, images_np: List[np.ndarray], settings: dict = None):
        # RGB to BGR
        images_np = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np]
        # prepare prompt
        if self.model_type == "text/visual":
            mode = settings["mode"]
            if mode == "text":
                class_names = settings["class_names"]
                self.classes = class_names
                self.model.set_classes(class_names, self.model.get_text_pe(class_names))
                predictions = self.model(
                    source=images_np,
                    conf=settings["conf"],
                )
            elif mode == "reference_image":
                reference_image_id = settings["reference_image_id"]
                if reference_image_id != self.reference_image_id:
                    self.reference_image = self._api.image.download_np(
                        reference_image_id
                    )
                    self.reference_image_id = reference_image_id
                class_names = [settings["reference_class_name"]]
                self.classes = class_names
                reference_bbox = settings["reference_bbox"]
                visual_prompts = dict(
                    bboxes=np.array(
                        [
                            [
                                reference_bbox[1],
                                reference_bbox[0],
                                reference_bbox[3],
                                reference_bbox[2],
                            ]
                        ],
                        dtype=np.float64,
                    ),
                    cls=np.array([0], dtype=np.int32),
                )
                predictions = self.model.predict(
                    images_np,
                    refer_image=self.reference_image,
                    visual_prompts=visual_prompts,
                    predictor=YOLOEVPSegPredictor,
                )

        else:
            predictions = self.model(
                source=images_np,
                conf=settings["conf"],
            )
        n = len(predictions)
        first_benchmark = predictions[0].speed
        # YOLO returns avg time per image, so we need to multiply it by the number of images
        benchmark = {
            "preprocess": first_benchmark["preprocess"] * n,
            "inference": first_benchmark["inference"] * n,
            "postprocess": first_benchmark["postprocess"] * n,
        }
        with sly.nn.inference.Timer() as timer:
            predictions = [
                self._to_dto(prediction, settings) for prediction in predictions
            ]
        to_dto_time = timer.get_time()
        benchmark["postprocess"] += to_dto_time
        return predictions, benchmark

    def _load_model_headless(
        self,
        model_files: dict,
        model_source: str,
        model_info: dict,
        device: str,
        runtime: str,
        **kwargs,
    ):
        """
        Diff to :class:`Inference`:
           - _set_model_meta_from_classes() removed due to lack of classes
        """
        deploy_params = {
            "model_files": model_files,
            "model_source": model_source,
            "model_info": model_info,
            "device": device,
            "runtime": runtime,
            **kwargs,
        }
        if model_source == ModelSource.CUSTOM:
            self._set_model_meta_custom_model(model_info)
            self._set_checkpoint_info_custom_model(deploy_params)
        self._load_model(deploy_params)
        # self._model_meta = sly.ProjectMeta()

    def _load_model(self, deploy_params: dict):
        """
        Diff to :class:`Inference`:
           - self.model_precision replaced with the cuda availability check
        """
        self.model_source = deploy_params.get("model_source")
        self.device = deploy_params.get("device")
        self.runtime = deploy_params.get("runtime", RuntimeType.PYTORCH)
        self.model_precision = torch.float32
        self._hardware = get_hardware_info(self.device)
        self.load_model(**deploy_params)
        self._model_served = True
        self._deploy_params = deploy_params
        if self.gui is not None:
            self.update_gui(self._model_served)
            self.gui.show_deployed_model_info(self)

    def set_project_meta(self, inference):
        """The model does not have predefined classes.
        In case of prompt-based models, the classes are defined by the user."""
        self.gui._model_classes_widget_container.hide()
        return

    def _create_label(self, dto: PredictionBBox) -> sly.Label:
        """
        Create a label from the prediction DTO.
        Diff to :class:`ObjectDetection`:
              - class_name is appended with "_bbox" to match the class name in the project
        """
        class_name = dto.class_name
        obj_class = self.model_meta.get_obj_class(class_name)
        if obj_class is None:
            self._model_meta = self.model_meta.add_obj_class(
                sly.ObjClass(class_name, sly.Rectangle)
            )
            obj_class = self.model_meta.get_obj_class(class_name)
        geometry = sly.Rectangle(*dto.bbox_tlbr)
        tags = []
        if dto.score is not None:
            tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
        label = sly.Label(geometry, obj_class, tags)
        return label

    def _to_dto(self, prediction, settings: dict) -> List[PredictionBBox]:
        """Converts YOLO Results to a List of Prediction DTOs."""
        dtos = []
        boxes_data = prediction.boxes.data
        for box in boxes_data:
            left, top, right, bottom, confidence, cls_index = (
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                float(box[4]),
                int(box[5]),
            )
            bbox = [top, left, bottom, right]
            dtos.append(PredictionBBox(self.classes[cls_index], bbox, confidence))
        return dtos
