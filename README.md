# **Saliency-Aware Language Guided Network for High-Resolution Salient Object Detection**
## 👉 Abstract
High-Resolution Salient Object Detection (HRSOD) aims to accurately identify and segment the most attention-grabbing targets in high-resolution images. Existing methods primarily rely on purely visual features for global saliency localization, lacking discriminative semantic features that can fundamentally distinguish targets. This limitation makes them susceptible to interference from non-salient objects in complex scenes, resulting in false positives and missed detections. To address this challenge, we propose a Saliency-Aware Language Guided Network (SALNet), which effectively leverages saliency priors in language to guide object recognition within the visual space. To further support cross-modal modeling, we construct L-HRSOD, the first cross-modal Vision-Language (VL) dataset specifically designed for HRSOD, consisting of 18,522 pairs of highly aligned high-resolution images and their corresponding language descriptions. Moreover, with the proposed dynamic language update strategy and multi-granularity Saliency-Aware attention module, our method facilitates deep information fusion in the joint VL representation space. This effectively suppresses non-salient interference, focuses on the salient targets described by the language, and recovers missing salient regions, significantly reducing false positives and missed detections. Experiments on existing HRSOD benchmark datasets show that SALNet significantly outperforms current vision-only methods in localization accuracy and multi-target recognition performance, validating the effectiveness of language guidance. See here for the relevant code and data.
## 👉 Network
<p align="center">
    <img width="1000" alt="image" src="https://anonymous.4open.science/r/UniHRSOD-7373/img/Network.png">
</p>
We have open-sourced the inference code and SALNet model weights. If you find any bugs due to carelessness on our part in organizing the code, feel free to contact us and point that!

## 👉 Installation
```sh
conda create -n SALNet python=3.8
conda activate SALNet
pip install -r requirements.txt
cd ops & sh make.sh # compile deformable attention
```

## 👉 Model Checkpoints

We host our model checkpoints and other pre-trained backbone weights on Google Netdisk,
Please download the checkpoint from [here](https://drive.google.com/drive/folders/1pOhwo3PCJO6Qy0atcjN-vwjNra_E2-X_?usp=drive_link) and place them under **"checkpoints/"**.

## 👉 Inference with Language Instruction
You can run the inference with Language Instruction by the following command:
```sh
python inference.py --image_path <IMG_PATH> --save_path <OUTPUT_PATH> --language_ins <'EXPRESSION'> 
```
- IMG_PATH: By default, the image is placed in the img folder
- language_ins: The default template is "the most salient object(s) in the image is(are) {}", with a description of the salient object in brackets, which can include details such as shape, color, and location.
- save_path: By default, the inference results are stored in the ./out directory.


