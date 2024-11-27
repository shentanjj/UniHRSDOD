from typing import List, Union
import torch.nn.functional as F
import argparse
from utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
import utils.config as config
from model import build_segmenter
import torchvision.transforms as transforms
from dataloader_collect.custom_transforms import *
# 初始化tokenizer
_tokenizer = _Tokenizer()

# 获取命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="单张图片推理")
    parser.add_argument('--image_path', type=str, default='./187179248_103d2db5ef_o.jpg',
                        help="输入图片的路径（默认：default_image.jpg）")
    parser.add_argument('--save_path', type=str, default='./output', help="保存输出图片的路径（默认：./output）")
    parser.add_argument('--text', type=str, default='the salient object in this image are a person and a koala',
                        help="输入图片的文本描述（默认：'A sample description of the image'）")
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file('config/train_test.yaml')
    cfg.image_path = args.image_path
    cfg.save_path = args.save_path
    cfg.text = args.text

    return cfg

# 文本tokenize函数
def tokenize(texts: Union[str, List[str]], context_length: int = 100, truncate: bool = False) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"输入的文本 {texts[i]} 超过了最大上下文长度 {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

# 将预测结果转换为numpy数组
def to_numpy(pred, shape):
    pred = F.interpolate(pred, shape, mode='bilinear', align_corners=True)
    pred = pred.data.cpu()
    pred = pred.numpy().squeeze()
    return pred

def get_transform(tfs):
    comp = []
    for key, value in zip(tfs.keys(), tfs.values()):
        if value is not None:
            tf = eval(key)(**value)
        else:
            tf = eval(key)()
        comp.append(tf)
    return transforms.Compose(comp)

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型配置
    model = build_segmenter(args).to(device)  # 传递字典cfg而不是args
    model_dict = torch.load(os.path.join(args.Test.checkpoint_dir, 'UniHRSOD_SFE.pth'))
    state_dict = model_dict['model_state_dict']
    for name, param in state_dict.items():
        print(f"{name},{param.shape}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 获取图像预处理转换
    transform = get_transform(args.Test.transforms)

    # 打开并预处理输入图片
    image = Image.open(args.image_path).convert('RGB')

    # 处理单张图片时，gt是None
    sample = {'image': image, 'text': args.text, 'ori_shape': image.size,
              'name': os.path.basename(args.image_path)}

    # 这里对gt进行处理时，transform会跳过gt的处理
    sample = transform(sample)

    # 提取处理后的图像
    inputs_v = sample['image'].unsqueeze(0).float().to(device)  # 将图像转为tensor并加上batch维度

    # 对输入文本进行tokenize
    text_tokens = tokenize(args.text, args.Model.word_len, True).to(device)

    # 推理
    with torch.no_grad():
        pre = model(inputs_v, text_tokens)
    pred = torch.sigmoid(pre[-1])

    # 将预测结果转为numpy数组
    pred = to_numpy(pred, inputs_v.shape[2:])

    # 创建保存路径文件夹（如果不存在）
    os.makedirs(args.save_path, exist_ok=True)

    # 获取原始图像的大小
    original_size = sample['ori_shape']  # (width, height)

    # 将预测结果调整为原始图像大小
    pred_image = Image.fromarray((pred * 255).astype(np.uint8))
    pred_image = pred_image.resize((original_size[0], original_size[1]), Image.BILINEAR)  # (width, height)

    # 获取原始图片的文件名并添加 .png 后缀
    output_image_name = os.path.splitext(sample['name'])[0] + '.png'

    # 保存调整后的预测结果
    pred_image.save(os.path.join(args.save_path, output_image_name))
    print(f"结果已保存至 {os.path.join(args.save_path, output_image_name)}")

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 使用合并后的 args 和 cfg 运行推理函数
    test(args)
