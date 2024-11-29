# **Unified High Resolution Salient Object Detection with Language Instruction**
## 👉 Abstract
High-Resolution Salient Object Detection (HRSOD) methods typically rely on purely visual features to detect and segment the most salient objects within an image. These models lack discriminative semantic features of salient objects, which makes them prone to false positives in complex scenes. To address this, we propose the Unified High-Resolution Salient Object Detection with Language Instruction(UniHRSOD) framework. First, we introduce a vision-language dataset L-HRSOD. To our knowledge, this is the first work to incorporate textual information into the HRSOD task. Based on L-HRSOD, we develop UniHRSOD, a language-driven, general-purpose HRSOD approach that integrates visual and language features. UniHRSOD uses a Fusion Attention Module (FAM) and a language update strategy to enhance its capability to recognize salient objects. Extensive experimental results indicate that UniHRSOD significantly outperforms existing purely visual methods, validating the effective guiding role of linguistic information.See here for the relevant code and data.
## 👉 Network
<p align="center">
    <img width="1000" alt="image" src="https://anonymous.4open.science/r/UniHRSOD-7373/img/Network.png">
</p>
We have open-sourced the inference code and UniHRSOD model weights. If you find any bugs due to carelessness on our part in organizing the code, feel free to contact us and point that!

## 👉 Installation
Install required packages.
```sh
conda create -n UniHRSOD python=3.8
conda activate UniLHRSOD
conda install pytorch==2.0.0 torchvision==2.0.0  torchaudio==2.0.0 -c pytorch -c conda-forge -y
pip install -r requirements.txt
```

## 👉 Model Checkpoints

We host our model checkpoints and other pre-trained backbone weights on Google Netdisk,
Please download the checkpoint from here and place them under **"checkpoints/"**.

## 👉 Inference with Language Instruction
You can run the inference with Language Instruction by the following command:
```sh
python inference.py --image_path <IMG_PATH> --save_path <OUTPUT_PATH> --language_ins <'EXPRESSION'> 
```
- IMG_PATH: By default, the image is placed in the img folder
- language_ins: The default template is "the most salient object(s) in the image is(are) {}", with a description of the salient object in brackets, which can include details such as shape, color, and location.
- save_path: By default, the inference results are stored in the ./out directory.


