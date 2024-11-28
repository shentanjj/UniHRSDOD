import os
import torch
import argparse
import numpy as np
from PIL import Image
import utils.config as config
from model import build_segmenter
from utils.config import apply_crf,to_numpy,get_transform,tokenize

# Get command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Single image inference")
    parser.add_argument('--image_path', type=str, default='./img/000065.jpg',
                        help="Input image path")
    parser.add_argument('--save_path', type=str, default='./output', help="Output image directory")
    parser.add_argument('--text', type=str, default='the most salient object in the image is a windmill,whose blades have a grid-like structure.',
                        help="Language description of the input image")
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file('config/train_test.yaml')
    cfg.image_path = args.image_path
    cfg.save_path = args.save_path
    cfg.text = args.text

    return cfg

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model configuration
    model = build_segmenter(args).to(device)
    model_dict = torch.load(os.path.join(args.Test.checkpoint_dir, 'UniHRSOD_SFE.pth'))
    state_dict = model_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = get_transform(args.Test.transforms)
    image = Image.open(args.image_path).convert('RGB')

    sample = {'image': image, 'text': args.text, 'ori_shape': image.size,
              'name': os.path.basename(args.image_path)}

    sample = transform(sample)

    inputs_v = sample['image'].unsqueeze(0).float().to(device)

    # Tokenize the input text
    text_tokens = tokenize(args.text, args.Model.word_len, True).to(device)

    # Inference
    with torch.no_grad():
        pre = model(inputs_v, text_tokens)
    pred = torch.sigmoid(pre[-1])

    # Convert the prediction results into numpy array
    pred = to_numpy(pred, inputs_v.shape[2:])
    os.makedirs(args.save_path, exist_ok=True)

    # Get the size of the original image
    original_size = sample['ori_shape']  # (width, height)

    # Resize the prediction result to the original image size
    pred_image = Image.fromarray((pred * 255).astype(np.uint8))
    pred_image = pred_image.resize((original_size[0], original_size[1]), Image.BILINEAR)

    pred_image = apply_crf(image, pred_image, num_classes=2)
    output_image_name = os.path.splitext(sample['name'])[0] + '.png'

    pred_image = Image.fromarray(pred_image.astype(np.uint8))
    pred_image.save(os.path.join(args.save_path, output_image_name))
    print(f"Results saved to {os.path.join(args.save_path, output_image_name)}")


if __name__ == "__main__":
    # Parsing command line arguments
    args = parse_args()

    # Run the inference function using the merged args and cfg
    test(args)