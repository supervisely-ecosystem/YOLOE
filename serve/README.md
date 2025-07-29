<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/YOLOE/releases/download/v0.0.1/poster_serve-yoloe.jpg"/>  

# Serve YOLOE

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/YOLOE/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/YOLOE)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/YOLOE/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/YOLOE/serve.png)](https://supervisely.com)

</div>

# Overview

YOLOE (Real-Time Seeing Anything) is a new advancement in zero-shot, promptable YOLO models, designed for open-vocabulary detection and segmentation. Unlike previous YOLO models limited to fixed categories, YOLOE uses text, image, or internal vocabulary prompts, enabling real-time detection of any object class. Built upon YOLOv10 and inspired by YOLO-World, YOLOE achieves state-of-the-art zero-shot performance with minimal impact on speed and accuracy.

![yoloe architecture](https://github.com/supervisely-ecosystem/YOLOE/releases/download/v0.0.1/yoloe_architecture.png)

YOLOE retains the standard YOLO structure —a convolutional backbone (e.g., CSP-Darknet) for feature extraction, a neck (e.g., PAN-FPN) for multi-scale fusion, and an anchor-free, decoupled detection head (as in YOLOv8/YOLO11) predicting objectness, classes, and boxes independently. YOLOE introduces three novel modules enabling open-vocabulary detection:

- Re-parameterizable Region-Text Alignment (RepRTA): Supports text-prompted detection by refining text embeddings (e.g., from CLIP) via a small auxiliary network. At inference, this network is folded into the main model, ensuring zero overhead. YOLOE thus detects arbitrary text-labeled objects (e.g., unseen "traffic light") without runtime penalties.

- Semantic-Activated Visual Prompt Encoder (SAVPE): Enables visual-prompted detection via a lightweight embedding branch. Given a reference image, SAVPE encodes semantic and activation features, conditioning the model to detect visually similar objects—a one-shot detection capability useful for logos or specific parts.

Lazy Region-Prompt Contrast (LRPC): In prompt-free mode, YOLOE performs open-set recognition using internal embeddings trained on large vocabularies (1200+ categories from LVIS and Objects365). Without external prompts or encoders, YOLOE identifies objects via embedding similarity lookup, efficiently handling large label spaces at inference.

# How To Run

**Step 1.** Select pretrained model architecture and press the **Serve** button

![pretrained_models](https://github.com/supervisely-ecosystem/YOLOE/releases/download/v0.0.1/yoloe_serve_0.png)

**Step 2.** Wait for the model to deploy

![deployed](https://github.com/supervisely-ecosystem/YOLOE/releases/download/v0.0.1/yoloe_serve_1.png)

# Acknowledgment

This app is based on the great work [YOLOE](https://github.com/ultralytics/ultralytics).