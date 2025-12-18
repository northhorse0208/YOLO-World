#这个脚本是用来记录lvis中的所有类别的稀有度，以及他在train，val，minival中的image_count值，这个image_count是直接根据类别中的image_count属性直接读取得到的。所以val和minival中每个类别是一致的
import json
import pandas as pd
import os


def load_category_info(json_path, split_name):
    """
    读取 JSON 文件中的 categories 信息
    返回一个字典: {category_id: image_count}
    对于 train 集，还会额外返回 name 和 frequency 信息
    """
    print(f"正在加载 {split_name} 文件: {json_path} ...")

    if not os.path.exists(json_path):
        print(f"警告: 文件不存在 -> {json_path}")
        return {}

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取该 split 下每个类别的 image_count
    # 结构: {id: count}
    count_map = {cat['id']: cat['image_count'] for cat in data['categories']}

    # 如果是训练集，我们需要更多元数据 (name, frequency)
    if split_name == 'train':
        meta_map = {}
        for cat in data['categories']:
            meta_map[cat['id']] = {
                'name': cat['name'],
                'frequency': cat['frequency'],  # r, c, f
                'train_count': cat['image_count']
            }
        return meta_map

    return count_map


def main():
    # ================= 配置路径 =================
    # 请根据你实际的文件路径修改这里
    train_json_path = '/data/codes/WangShuo/dataset/coco/annotations/lvis_v1_train.json'
    val_json_path = '/data/codes/WangShuo/dataset/coco/annotations/lvis_v1_val.json'
    minival_json_path = '/data/codes/WangShuo/dataset/coco/annotations/lvis_v1_minival_inserted_image_name.json'
    # ===========================================

    # 1. 以 Train 为主表构建基础数据 (因为 Train 包含最全的类别和 frequency 定义)
    # 结果结构: {id: {'name':Str, 'frequency':Str, 'train_count':Int}}
    combined_data = load_category_info(train_json_path, 'train')

    if not combined_data:
        print("错误: 无法加载训练集数据，脚本终止。")
        return

    # 2. 加载 Val 数据
    val_counts = load_category_info(val_json_path, 'val')

    # 3. 加载 Minival 数据
    minival_counts = load_category_info(minival_json_path, 'minival')

    # 4. 合并数据
    print("正在合并数据...")
    final_rows = []

    # LVIS 的类别 ID 通常是 1 到 1203
    for cat_id, info in combined_data.items():
        row = {
            'Category ID': cat_id,
            'Name': info['name'],
            'Frequency': info['frequency'],  # r=Rare, c=Common, f=Frequent
            'Train Count': info['train_count'],
            'Val Count': val_counts.get(cat_id, 0),  # 如果Val里没有该类，记为0
            'Minival Count': minival_counts.get(cat_id, 0)  # 如果Minival里没有该类，记为0
        }
        final_rows.append(row)

    # 5. 转换为 DataFrame 并展示/保存
    df = pd.DataFrame(final_rows)

    # 按照 Train Count 降序排列 (可选)
    df = df.sort_values(by='Train Count', ascending=False)

    # 打印前 10 行预览
    print("\n数据预览:")
    print(df.head(10))

    # 保存为 CSV
    output_file = 'lvis_category_stats.csv'
    df.to_csv(output_file, index=False)
    print(f"\n统计完成！结果已保存至: {output_file}")

    # 简单的统计分析
    print("\n=== 简单统计 ===")
    print(f"总类别数: {len(df)}")
    print(f"Rare 类别数: {len(df[df['Frequency'] == 'r'])}")
    print(f"Common 类别数: {len(df[df['Frequency'] == 'c'])}")
    print(f"Frequent 类别数: {len(df[df['Frequency'] == 'f'])}")


if __name__ == '__main__':
    main()