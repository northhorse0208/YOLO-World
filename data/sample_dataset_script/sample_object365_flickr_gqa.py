import json
import os
import random
import argparse
import shutil

from charset_normalizer.md import annotations
from tqdm import tqdm

'''

采样object365图片
python sample_object365_flickr_gqa.py --image_path ./../../../../datasets/Objects365v1/images/train --anno_path ./../../../../datasets/Objects365v1/annotations/objects365_train.json --sampling_ratio 0.001 --output_dir ./../../../../datasets/Objects365v1/object365v1_sampled
采样flickr
python sample_object365_flickr_gqa.py --image_path './../../../../../dataset/flickr 30k/flickr30k-images' --anno_path './../../../../../dataset/flickr 30k/final_flickr_separateGT_train.json' --sampling_ratio 0.001 --output_dir './../../../../../dataset/flickr 30k/flickr_sampled'
采样gqa/goldg
python sample_object365_flickr_gqa.py --image_path './../../../../../dataset/GQA/images' --anno_path './../../../../../dataset/GQA/final_mixed_train_no_coco.json' --sampling_ratio 0.001 --output_dir './../../../../../dataset/GQA/gqa_sampled'

'''

def sample_coco_dataset(image_dir, anno_path, sampling_ratio, output_dir):

    """
    对 COCO 格式的数据集进行采样。

    参数:
    - image_dir (str): 原始图片所在的目录。
    - annot_path (str): 原始 COCO 格式标注文件的路径。
    - sampling_ratio (float): 采样比例，介于 0 和 1 之间。
    - output_dir (str): 输出采样后数据集的根目录。
    """

    if not 0 < sampling_ratio <= 1:
        raise ValueError("采样比例必须在(0, 1]之间")

    #1。创建目标目录结构
    sample_dataset_name = f'sampling_ratio_{sampling_ratio}'
    sample_base_path = os.path.join(output_dir, sample_dataset_name)
    sample_image_dir = os.path.join(sample_base_path, 'images')
    sample_annot_dir = os.path.join(sample_base_path, 'annotations')

    print(f'创建采样数据集目录{sample_base_path} ...........')
    os.makedirs(sample_image_dir, exist_ok=True)
    os.makedirs(sample_annot_dir, exist_ok=True)

    #2. 加载原始标注文件
    print(f'正在从{anno_path}加载标注文件，请稍候...')
    with open(anno_path, 'r') as f:
        coco_data = json.load(f)

    #3. 对图片信息进行采样
    images_info = coco_data['images']
    num_total_images = len(images_info)
    num_sampled_images = int(num_total_images * sampling_ratio)

    if num_sampled_images == 0:
        print('警告：采样比例过低，没有图片被采样。请尝试提高采样比例')
        return

    print(f"共计{num_total_images}张图片，将随机采样{num_sampled_images}张图片，采样比例为{sampling_ratio}")

    #随机抽取图片
    sampled_images = random.sample(images_info, num_sampled_images)

    #为了快速查找，创建一个包含所有被采样图片ID的集合
    sampled_image_ids = {img['id'] for img in sampled_images}

    # 4. 根据采样的图片筛选对应的标注信息
    print("正在筛选与采样图片对应的标注信息...")
    all_annotations = coco_data.get('annotations', [])
    sampled_annotations = [
        anno for anno in all_annotations
        if anno['image_id'] in sampled_image_ids
    ]

    # 5. 构建新的 COCO 格式字典
    new_coco_data = {
        'info' : coco_data.get('info', []),
        'licenses' : coco_data.get('license', []),
        'images' : sampled_images,
        'annotations' : sampled_annotations,
        'categories' : coco_data.get('categories', [])
    }

    # 6. 保存新的标注文件
    # 从原始标注文件名中提取基本名称，并添加后缀
    original_annot_filename = os.path.basename(anno_path)
    base_name, ext = os.path.splitext(original_annot_filename)
    new_annot_filename = f"{base_name}_sampled_{sampling_ratio:.2%}{ext}"
    new_annot_path = os.path.join(sample_annot_dir, new_annot_filename)
    print(f'正在将新的标注文件写入：{new_annot_path}')
    with open(new_annot_path, 'w') as f:
        json.dump(new_coco_data, f)


    # 7. 复制采样后的图片文件
    print('正在复制采样后的图片')
    for img_info in tqdm(sampled_images):
        file_name = img_info['file_name']

        original_path = os.path.join(image_dir, file_name)
        destination_path = os.path.join(sample_image_dir, os.path.basename(file_name))

        if os.path.exists(original_path):
            if not os.path.exists(destination_path):
                shutil.copy(original_path, destination_path)
        else:
            print(f'警告：图片文件{original_path}未找到，已跳过')

    print('\n 数据集采样完成')
    print('=' * 30)
    print(f'采样后的图片存放于{sample_image_dir}')
    print(f'采样后的标注文件存放于{new_annot_path}')
    print('=' * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="对coco格式的数据集进行采样，object365也是coco格式的")
    parser.add_argument('--image_path', type=str, required=True,
                        help='原始图片所在根目录')
    parser.add_argument('--anno_path', type=str, required=True,
                        help='原始coco格式标注文件的完整路径')
    parser.add_argument('--sampling_ratio', type=float, required=True,
                        help='采样比例，例如0.01代表采样1%')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='用于存放生成采样数据集的目录')

    args = parser.parse_args()

    sample_coco_dataset(image_dir=args.image_path,
                        anno_path=args.anno_path,
                        sampling_ratio=args.sampling_ratio,
                        output_dir=args.output_dir)
