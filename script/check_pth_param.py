import torch
import numpy as np


def check_model_diff(model_path_a, model_path_b):
    print(f"Loading model A: {model_path_a}")
    print(f"Loading model B: {model_path_b}")

    # 加载模型，处理 mmdet checkpoint 包含 meta data 的情况
    ckpt_a = torch.load(model_path_a, map_location='cpu')
    ckpt_b = torch.load(model_path_b, map_location='cpu')

    # 通常权重在 'state_dict' 键下，如果没有则直接使用 ckpt
    state_dict_a = ckpt_a.get('state_dict', ckpt_a)
    state_dict_b = ckpt_b.get('state_dict', ckpt_b)

    # 统计数据
    stats = {
        'backbone': {'changed': 0, 'unchanged': 0},
        'neck': {'changed': 0, 'unchanged': 0},
        'head': {'changed': 0, 'unchanged': 0},
        'other': {'changed': 0, 'unchanged': 0}  # 比如 rpn 或其他组件
    }

    # 获取所有键的交集
    keys_a = set(state_dict_a.keys())
    keys_b = set(state_dict_b.keys())
    common_keys = keys_a.intersection(keys_b)

    print(f"\nComparing {len(common_keys)} common parameters...\n")

    for key in common_keys:
        # 获取参数张量
        param_a = state_dict_a[key]
        param_b = state_dict_b[key]

        # 判断参数是否相等
        # 使用 equal 严格判断，或者使用 allclose 允许微小误差
        is_same = torch.equal(param_a, param_b)

        # 根据 key 的名字归类 (MMDetection 命名惯例)
        if 'backbone' in key:
            category = 'backbone'
        elif 'neck' in key:
            category = 'neck'
        elif 'head' in key or 'bbox_head' in key or 'mask_head' in key:
            category = 'head'
        else:
            category = 'other'

        if is_same:
            stats[category]['unchanged'] += 1
        else:
            stats[category]['changed'] += 1
            # 只有当你想看具体哪里变了的时候取消下面的注释
            # print(f"Diff found in: {key}")

    # 打印总结报告
    print("-" * 60)
    print(f"{'Module':<15} | {'Changed Params':<15} | {'Unchanged Params':<15}")
    print("-" * 60)

    for cat, data in stats.items():
        print(f"{cat:<15} | {data['changed']:<15} | {data['unchanged']:<15}")
    print("-" * 60)

    # 简单的结论判断
    if stats['backbone']['changed'] == 0 and stats['head']['changed'] > 0:
        print("\n✅ 结论: 看起来很成功！Backbone 被冻结了，只有 Head 发生了变化。")
    elif stats['backbone']['changed'] > 0:
        print("\n⚠️ 警告: Backbone 的参数发生了变化。如果没有解冻 Backbone，可能是 BatchNorm 没冻结。")
    else:
        print("\nℹ️ 结果分析请参考上表。")


# ================= 使用方法 =================
# 替换成你实际的文件路径
#path_base = '/data/codes/WangShuo/py_project/YOLO-World-research/YOLO-World/work_dirs/finetune_deformable_contrast_fuse_fromofficialyoloworld_val_minival/epoch_2.pth'  # 或者预训练权重
path_base = '/data/codes/WangShuo/py_project/YOLO-World-research/YOLO-World/official_pretraind_models/yolo-world-l-640.pth'
path_tuned = '/data/codes/WangShuo/py_project/YOLO-World-research/YOLO-World/work_dirs/finetune_deformable_contrast_fuse_head0.1_val_minival/20251210_194641/epoch_10.pth'  # 微调后的权重

check_model_diff(path_base, path_tuned)