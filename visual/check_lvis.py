import json
import os
from collections import defaultdict


def verify_lvis_integrity(json_path, split_name):
    print(f"--- 正在检查: {split_name} ({os.path.basename(json_path)}) ---")

    if not os.path.exists(json_path):
        print(f"错误: 文件不存在 -> {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"  > 图片总数 (Images): {len(data['images'])}")
    print(f"  > 标注总数 (Annotations): {len(data['annotations'])}")
    print(f"  > 类别定义数 (Categories): {len(data['categories'])}")

    # 1. 提取元数据中的“声称值”
    claimed_counts = {cat['id']: cat.get('image_count', 0) for cat in data['categories']}
    category_names = {cat['id']: cat['name'] for cat in data['categories']}

    # 2. 计算“真实值”
    # 逻辑: 遍历所有标注，记录每个 category_id 对应的 unique image_id
    real_counts_map = defaultdict(set)
    for ann in data['annotations']:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        real_counts_map[cat_id].add(img_id)

    # 转换 set 长度为 int
    real_counts = {cat_id: len(img_ids) for cat_id, img_ids in real_counts_map.items()}

    # 3. 对比寻找差异
    mismatch_count = 0
    mismatch_examples = []

    for cat_id, claimed in claimed_counts.items():
        actual = real_counts.get(cat_id, 0)  # 如果没有标注，真实值就是0

        if claimed != actual:
            mismatch_count += 1
            if len(mismatch_examples) < 5:  # 只记录前5个例子用于展示
                mismatch_examples.append({
                    'name': category_names.get(cat_id, 'Unknown'),
                    'id': cat_id,
                    'claimed': claimed,
                    'actual': actual
                })

    # 4. 输出诊断结果
    if mismatch_count == 0:
        print(f"✅ 结果: {split_name} 数据完全一致！元数据是正确的。")
    else:
        print(f"❌ 结果: {split_name} 发现异常！")
        print(f"  > 共有 {mismatch_count} 个类别的 image_count 与实际标注不符。")
        print("  > 差异示例 (前5个):")
        for ex in mismatch_examples:
            print(f"    - {ex['name']} (ID: {ex['id']}): 标称={ex['claimed']}, 实际={ex['actual']}")

    print("\n")


def main():
    # ================= 修改这里 =================
    val_path = '/data/codes/WangShuo/dataset/coco/annotations/lvis_v1_val.json'
    minival_path = '/data/codes/WangShuo/dataset/coco/annotations/lvis_v1_minival_inserted_image_name.json'
    # ===========================================

    verify_lvis_integrity(val_path, "Validation Set")
    verify_lvis_integrity(minival_path, "Minival Set")


if __name__ == "__main__":
    main()