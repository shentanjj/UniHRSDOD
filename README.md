## 👉 Installation
```sh
Install required packages.
conda create -n UniHRSOD python=3.8
conda activate UniLHRSOD
conda install pytorch==2.0.0 torchvision==2.0.0  torchaudio==2.0.0 -c pytorch -c conda-forge -y
pip install -r requirements.txt
```

## 👉 Model Checkpoints

We host our model checkpoints and other pre-trained backbone weights on Google Netdisk: https://pan.baidu.com/s/1eCuHs3qhd1lyVGqUOdaeFw?pwd=r1pg, Password：r1pg 

Please download the checkpoint from Google Netdisk and place them under **"ckpt/"**.

## 👉 Inference with Language Instruction
You can run the general inference by the following command:
```sh
python inference.py --image_path <IMG_PATH> --save_path <OUTPUT_PATH>--language_ins <'EXPRESSION'> 

python general_inference.py  --img <IMG_PATH> --exp <'EXPRESSION'> --sp <MASK_SAVE_PATH>
```
- IMG_PATH: By default, the image is placed in the img folder
- language_ins: The default template is "the most salient object(s) in the image is(are) {}", with a description of the salient object in brackets, which can include details such as shape, color, and location.
- save_path: By default, the inference results are stored in the ./out directory.


