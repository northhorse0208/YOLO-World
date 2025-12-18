import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lvis import LVIS

# ================= 配置路径 =================
DATA_PATH = "debug_opr_data.pt"
# 请修改为你服务器上的 LVIS 验证集或训练集 json 路径
# 我们需要它来获取每个类别的 image_count 以区分 R/C/F
LVIS_JSON_PATH = "/data/codes/WangShuo/dataset/coco/annotations/lvis_v1_minival_inserted_image_name.json"


# ===========================================

def analyze():
    print(f"Loading data from {DATA_PATH}...")
    try:
        data = torch.load(DATA_PATH)
        # gate shape: (1203, 512) -> mean -> (1203,)
        gate = data['gate'].numpy()
        gate_mean_per_class = gate.mean(axis=1)
        print(f"Gate tensor shape: {gate.shape}")
    except FileNotFoundError:
        print("❌ Error: debug_opr_data.pt not found. Please run the debugger code first.")
        return

    print(f"Loading LVIS annotations from {LVIS_JSON_PATH}...")
    try:
        # 尝试使用 LVIS API 加载，如果没有安装 lvis 库，则回退到普通 json 加载
        lvis_api = LVIS(LVIS_JSON_PATH)
        cats = lvis_api.dataset['categories']
    except (ImportError, FileNotFoundError):
        print("⚠️ 'lvis' library not found or file missing, trying manual json load...")
        with open(LVIS_JSON_PATH, 'r') as f:
            dataset = json.load(f)
            cats = dataset['categories']

    # === 关键步骤：对齐 Tensor ===
    # YOLO-World/MMDet 通常按照 category_id 的升序排列分类器权重
    # 我们必须确保 JSON 里的类别顺序与 gate_mean_per_class 的行顺序一致
    sorted_cats = sorted(cats, key=lambda x: x['id'])

    if len(sorted_cats) != len(gate_mean_per_class):
        print(
            f"⚠️ Warning: LVIS categories count ({len(sorted_cats)}) != Gate tensor rows ({len(gate_mean_per_class)})")
        print("Please ensure you are using the correct json file corresponding to the model's class head.")
        return

    # 提取属性
    freq_tags = [c.get('frequency', 'u') for c in sorted_cats]  # 'r', 'c', 'f'
    img_counts = [c.get('image_count', 0) for c in sorted_cats]

    # 映射全称
    freq_map = {'r': 'Rare', 'c': 'Common', 'f': 'Frequent', 'u': 'Unknown'}

    # === 1. 数据分组 (Based on Official Frequency Attribute) ===
    groups = {'Rare': [], 'Common': [], 'Frequent': []}

    # 用于散点图的数据
    scatter_data = []  # list of (count, gate_val, group_name)

    for idx, tag in enumerate(freq_tags):
        val = gate_mean_per_class[idx]
        count = img_counts[idx]
        group_name = freq_map.get(tag, 'Unknown')

        if group_name in groups:
            groups[group_name].append(val)
            scatter_data.append({'count': count, 'gate': val, 'group': group_name})

    # === 2. 绘制箱线图 (Boxplot) ===
    plt.figure(figsize=(10, 6))

    # 准备 Seaborn 数据格式
    plot_x = []
    plot_y = []

    # 强制顺序: Rare -> Common -> Frequent
    for group_name in ['Rare', 'Common', 'Frequent']:
        values = groups[group_name]
        if not values: continue
        plot_x.extend([group_name] * len(values))
        plot_y.extend(values)
        print(f"{group_name}: Mean Gate = {np.mean(values):.4f}, Count = {len(values)}")

    sns.boxplot(x=plot_x, y=plot_y, palette="Set2", showfliers=False)  # showfliers=False 隐藏离群点让图更清晰
    sns.stripplot(x=plot_x, y=plot_y, color='black', alpha=0.3, size=2, jitter=True)  # 加上散点看密度

    plt.title("Gate Value Distribution by LVIS Frequency Tag\n(Official 'r'/'c'/'f' Split)")
    plt.ylabel("Average Gate Value (0-1)")
    plt.grid(True, axis='y', alpha=0.3)

    out_box = "gate_analysis_boxplot_official.png"
    plt.savefig(out_box, dpi=150)
    print(f"Saved boxplot to {out_box}")

    # === 3. 绘制散点图 (Gate vs Instance Count colored by Frequency) ===
    plt.figure(figsize=(10, 6))

    # 将数据解包以便绘图
    # 我们依然使用 image_count 作为 X 轴，因为这能展示长尾分布的物理特性
    # 但颜色由 official frequency tag 决定
    colors = {'Rare': 'red', 'Common': 'blue', 'Frequent': 'green'}

    for group_name in ['Rare', 'Common', 'Frequent']:
        subset = [d for d in scatter_data if d['group'] == group_name]
        if not subset: continue

        x_vals = [d['count'] for d in subset]
        y_vals = [d['gate'] for d in subset]

        plt.scatter(x_vals, y_vals, c=colors[group_name], label=group_name, alpha=0.6, s=15, edgecolors='none')

    plt.xscale('log')
    plt.xlabel("Number of Training Images (Log Scale)")
    plt.ylabel("Average Gate Value")
    plt.title("Gate Value vs. Instance Count (Colored by Official Split)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_scatter = "gate_analysis_scatter_official.png"
    plt.savefig(out_scatter, dpi=150)
    print(f"Saved scatter plot to {out_scatter}")


if __name__ == "__main__":
    analyze()