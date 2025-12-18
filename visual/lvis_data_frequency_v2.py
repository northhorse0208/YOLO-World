#
#这个脚本是用来记录lvis中的所有类别的稀有度，以及他在train，val，minival中的image_count值，
# 这个image_count是遍历 annotations -> category_id -> Set(image_id) -> len()得到的，所以更准确

import json
import pandas as pd
import os
from collections import defaultdict


def get_real_image_counts(json_path, split_name):
    """
    忽略 categories 中的 image_count 字段，
    直接遍历 annotations 计算真实的 image_count。
    """
    print(f"[{split_name}] 正在加载并计算真实统计: {json_path} ...")

    if not os.path.exists(json_path):
        print(f"⚠️ 警告: 文件不存在 -> {json_path}")
        return None, {}

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. 如果是训练集，我们需要提取类别的元数据 (ID -> Name, Frequency)
    #    注意：我们只信任 Train 的 name 和 frequency 定义
    meta_info = {}
    if split_name == 'train':
        for cat in data['categories']:
            meta_info[cat['id']] = {
                'name': cat['name'],
                'frequency': cat['frequency']  # r, c, f
            }

    # 2. 计算真实的 Image Count
    #    逻辑: category_id -> set(image_id)
    #    使用 set 是为了去重，因为一张图里可能有多个同类的物体
    cat_to_images = defaultdict(set)

    total_anns = len(data['annotations'])
    print(f"   > 正在扫描 {total_anns} 条标注...")

    for ann in data['annotations']:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        cat_to_images[cat_id].add(img_id)

    # 将 set 转换为数量 (int)
    real_counts = {cat_id: len(img_set) for cat_id, img_set in cat_to_images.items()}

    print(f"   > 计算完成，覆盖了 {len(real_counts)} 个类别。")

    return meta_info, real_counts


def main():
    # ================= 配置路径 =================
    # 请修改为你本地的真实路径
    train_json_path = '/data/codes/WangShuo/dataset/coco/annotations/lvis_v1_train.json'
    val_json_path = '/data/codes/WangShuo/dataset/coco/annotations/lvis_v1_val.json'
    minival_json_path = '/data/codes/WangShuo/dataset/coco/annotations/lvis_v1_minival_inserted_image_name.json'
    # ===========================================

    # 1. 处理 Train (作为主表，提供 Name, Frequency 和 Train Count)
    train_meta, train_real_counts = get_real_image_counts(train_json_path, 'train')

    if not train_meta:
        print("错误: 无法加载训练集元数据，程序终止。")
        return

    # 2. 处理 Val
    _, val_real_counts = get_real_image_counts(val_json_path, 'val')

    # 3. 处理 Minival
    _, minival_real_counts = get_real_image_counts(minival_json_path, 'minival')

    # 4. 合并数据
    print("\n正在合并所有统计数据...")
    rows = []

    # LVIS 标准类别 ID 范围通常是 1-1203，我们遍历 Train 中定义的所有类别
    # 这样即使某个类别在 Val/Minival 没出现，也能显示为 0
    for cat_id, meta in train_meta.items():
        # 获取各数据集的真实计数，如果没有则为 0
        c_train = train_real_counts.get(cat_id, 0)
        c_val = val_real_counts.get(cat_id, 0)
        c_minival = minival_real_counts.get(cat_id, 0)

        row = {
            'Category ID': cat_id,
            'Name': meta['name'],
            'Frequency': meta['frequency'],  # 这里的 r/c/f 是 LVIS 官方定义的属性
            'Train Count (Real)': c_train,
            'Val Count (Real)': c_val,
            'Minival Count (Real)': c_minival
        }
        rows.append(row)

    # 5. 生成 DataFrame 并保存
    df = pd.DataFrame(rows)

    # 按照 Train Count 降序排列
    df = df.sort_values(by='Train Count (Real)', ascending=False)

    # 保存结果
    output_file = 'lvis_real_stats_calculated.csv'
    df.to_csv(output_file, index=False)

    print(f"\n✅ 成功！CSV 文件已生成: {output_file}")
    print("-" * 30)
    print("数据预览 (Top 10):")
    print(df.head(10).to_string(index=False))

    # 简单验证 Minival 的稀疏性
    zero_in_minival = len(df[df['Minival Count (Real)'] == 0])
    print("-" * 30)
    print(f"统计摘要:")
    print(f"Minival 中计数为 0 的类别数量: {zero_in_minival} / {len(df)}")
    print(f"这意味着有 {zero_in_minival} 个类别在 Minival 评测时实际上没有样本参与计算。")


if __name__ == '__main__':
    main()