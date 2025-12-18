import numpy as np
from ultralytics.utils import yaml_load
from ultralytics.utils.torch_utils import smart_inference_mode
import torch
from tqdm import tqdm
import os
# from ultralytics.nn.text_model import build_text_model
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTextConfig

# @smart_inference_mode()
# def generate_label_embedding(model, texts, batch=512):
#     model = build_text_model(model, device='cuda')
#     assert(not model.training)
#
#     text_tokens = model.tokenize(texts)
#     txt_feats = []
#     for text_token in tqdm(text_tokens.split(batch)):
#         txt_feats.append(model.encode_text(text_token))
#     txt_feats = torch.cat(txt_feats, dim=0)
#     return txt_feats.cpu()

def generate_yolo_world_embedding(model_name, texts, batch_size=512, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clip_config = CLIPTextConfig.from_pretrained(model_name, attention_dropout=0.0)
    model = CLIPTextModelWithProjection.from_pretrained(model_name, config=clip_config)

    model.to(device)
    model.eval()

    all_feats = []
    print(f'Encoding{len(texts)}texts...')

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i+batch_size]

            inputs = tokenizer(
                text=batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=77
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            txt_feats = outputs.text_embeds

            txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
            all_feats.append(txt_feats.cpu())
        all_feats = torch.cat(all_feats, dim=0)
        return all_feats



def collect_grounding_labels(cache_path):
    labels = np.load(cache_path, allow_pickle=True)
    cat_names = set()
    
    for label in labels:
        for text in label["texts"]:
            for t in text:
                t = t.strip()
                assert(t)
                cat_names.add(t)
    
    return cat_names

def collect_detection_labels(yaml_path):
    cat_names = set()
    
    data = yaml_load(yaml_path, append_filename=True)
    names = [name.split("/") for name in data["names"].values()]
    for name in names:
        for n in name:
            n = n.strip()
            assert(n)
            cat_names.add(n)
    
    return cat_names


def collect_txt_labels(txt_path):
    """
    从纯文本文件中读取所有唯一的类别名称，并返回一个集合。

    Args:
        txt_path (str): 包含缺失文本的文件的路径（每行一个词）。

    Returns:
        set: 包含所有唯一文本名称的集合。
    """
    cat_names = set()

    if not os.path.exists(txt_path):
        print(f"Warning: Missing text file not found at {txt_path}. Skipping collection.")
        return cat_names  # 返回空集合，防止程序中断

    print(f"Collecting missing labels from {txt_path}...")

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 逐行读取文件
            for line in f:
                t = line.strip()  # 去除首尾空格和换行符
                if t:  # 确保文本非空
                    cat_names.add(t)
    except Exception as e:
        print(f"Error reading file {txt_path}: {e}")

    return cat_names

if __name__ == '__main__':
    os.environ["PYTHONHASHSEED"] = "0"
    
    flickr_cache = './../../../dataset/flickr 30k/final_flickr_separateGT_train_segm.cache'
    mixed_grounding_cache = './../../../dataset/GQA/final_mixed_train_no_coco_segm.cache'
    objects365v1_yaml = 'embedding_cache/yaml/Objects365v1.yaml'
    missing_text = 'embedding_cache/missing_txt_logs/final_missing_texts_clean.txt'
    
    all_cat_names = set()
    all_cat_names |= collect_detection_labels(objects365v1_yaml)
    all_cat_names |= collect_txt_labels(missing_text)
    all_cat_names |= collect_grounding_labels(flickr_cache)
    all_cat_names |= collect_grounding_labels(mixed_grounding_cache)

    
    all_cat_names = list(all_cat_names)

    #下面这两行是使用yoloe的mobileclip:blt 和 使用yoloworld 的clip提取文本张量，到时候可以具体选择用哪个
    #yoloe
    # model = yaml_load('ultralytics/cfg/default.yaml')['text_model']
    # all_cat_feats = generate_label_embedding(model, all_cat_names)
    #yolo-world
    model = 'openai/clip-vit-base-patch32'
    all_cat_feats = generate_yolo_world_embedding(model, texts=all_cat_names,batch_size=512)

    cat_name_feat_map = {}
    for name, feat in zip(all_cat_names, all_cat_feats):
        cat_name_feat_map[name] = feat
    
    os.makedirs(f'embedding_cache/{model}', exist_ok=True)
    torch.save(cat_name_feat_map, f'embedding_cache/{model}/train_label_embeddings.pt')
