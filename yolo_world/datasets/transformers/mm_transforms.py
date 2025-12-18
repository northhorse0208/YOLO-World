# Copyright (c) Tencent Inc. All rights reserved.
import json
import random
from typing import Tuple

import numpy as np
import torch
from mmyolo.registry import TRANSFORMS
import os
import torch.distributed as dist
from torch.utils.data import get_worker_info


@TRANSFORMS.register_module()
class RandomLoadText:

    def __init__(self,
                 text_path: str = None,
                 text_model_name: str = 'clip-vit-base-patch32',  # 指定模型目录名
                 tools_dir: str = 'embedding_cache/openai',  # 指定缓存文件所在的根目录
                 missing_log_path = 'embedding_cache/missing_txt_logs',
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

        cache_dir = f"{tools_dir}/{text_model_name.replace('/','_')}"
        embed_path = f'{cache_dir}/train_label_embeddings.pt'
        # 打印信息建议只在主进程打印，防止多卡刷屏
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Loading cached embeddings from {embed_path}......")
        self.train_label_embeddings = torch.load(embed_path, map_location='cpu')

        # 加载全局负样本池 (N, Dim)
        neg_embed_path = f'{cache_dir}/global_grounding_neg_embeddings.pt'
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Loading global negative embeddings from {neg_embed_path}...")
        self.global_grounding_neg_embeddings = torch.load(neg_embed_path, map_location='cpu')

        # 【修改 2】初始化日志目录
        self.missing_log_dir = missing_log_path
        os.makedirs(self.missing_log_dir, exist_ok=True)

        # 用于记录当前 worker 进程已经记录过的词
        self._logged_missing_keys = set()

    def _get_unique_log_path(self):
        """
        【新增辅助函数】动态获取当前进程唯一的日志文件路径
        命名格式: missing_rank{RANK}_worker{WORKER_ID}.txt
        """
        # 1. 获取 Rank (显卡编号)
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
        except RuntimeError:
            rank = 0

        # 2. 获取 Worker ID (DataLoader 进程编号)
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        # 3. 组合文件名
        filename = f"missing_rank{rank}_worker{worker_id}.txt"
        return os.path.join(self.missing_log_dir, filename)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        num_classes = len(class_texts)
        if 'gt_labels' in results:
            gt_label_tag = 'gt_labels'
        elif 'gt_bboxes_labels' in results:
            gt_label_tag = 'gt_bboxes_labels'
        else:
            raise ValueError('No valid labels found in results.')
        positive_labels = set(results[gt_label_tag])

        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        # yolo-world中将正样本和负样本的序列做成随机的了，下面我修改成了正样本类别号一定在前面
        # sampled_labels = list(positive_labels) + list(negative_labels)
        # random.shuffle(sampled_labels)

        pos_list = list(positive_labels)
        neg_list = list(negative_labels)
        random.shuffle(pos_list)
        random.shuffle(neg_list)
        sampled_labels = pos_list + neg_list

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['gt_bboxes']), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag]):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label]
        results['gt_bboxes'] = results['gt_bboxes'][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

        if 'instances' in results:
            retaged_instances = []
            for idx, inst in enumerate(results['instances']):
                label = inst['bbox_label']
                if label in label2ids:
                    inst['bbox_label'] = label2ids[label]
                    retaged_instances.append(inst)
            results['instances'] = retaged_instances

        # texts = []
        # for label in sampled_labels:
        #     cls_caps = class_texts[label]
        #     assert len(cls_caps) > 0
        #     cap_id = random.randrange(len(cls_caps))
        #     sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
        #     texts.append(sel_cls_cap)
        #
        # if self.padding_to_max:
        #     num_valid_labels = len(positive_labels) + len(negative_labels)
        #     num_padding = self.max_num_samples - num_valid_labels
        #     if num_padding > 0:
        #         texts += [self.padding_value] * num_padding
        #
        # results['texts'] = texts]

        txt_feats_list = []
        for label in sampled_labels:
            cls_caps = class_texts[label]
            assert len(cls_caps) > 0

            #随机选择一个描述
            cap_id = random.randrange(len(cls_caps))
            selected_text = cls_caps[cap_id]

            key = selected_text.strip()
            if key in self.train_label_embeddings:
                txt_feats_list.append(self.train_label_embeddings[key])
            else:
                # --- 缺失处理 ---
                embedding_dim = self.global_grounding_neg_embeddings.shape[1]
                zero_embedding = torch.zeros(embedding_dim)
                txt_feats_list.append(zero_embedding)

                # 【修改 3】多进程安全的写入逻辑
                # 先检查内存中的 set，避免同一进程重复 IO
                if key not in self._logged_missing_keys:
                    try:
                        # 动态获取当前进程专属的文件路径
                        log_file_path = self._get_unique_log_path()

                        # 追加写入
                        with open(log_file_path, 'a', encoding='utf-8') as f:
                            f.write(f"{key}\n")

                        # 标记为已记录
                        self._logged_missing_keys.add(key)

                        # 仅在主进程打印 Warning，防止日志爆炸
                        # (可选: 也可以全打印，看你调试需求)
                        # print(f"Warning: Missing key '{key}' logged to {log_file_path}")

                    except Exception as e:
                        # 捕获异常防止卡死训练
                        print(f"Error writing to log file: {e}")

        # 堆叠成 Tensor (N, Dim)
        if len(txt_feats_list) > 0:
            txt_feats = torch.stack(txt_feats_list, dim=0)
        else:
            # 极端情况：没有样本
            print(f"出现了该张图片中没有样本的情况")
            embedding_dim = self.global_grounding_neg_embeddings.shape[1]
            txt_feats = torch.zeros((0, embedding_dim))

        # ------------------------------------------------------
        # 核心修改：Padding (使用全局负样本池)
        # 这里使用了负样本，而不是像yoloworld直接使用空格做填补
        # ------------------------------------------------------
        if self.padding_to_max:
            num_valid_labels = txt_feats.shape[0]
            num_padding = self.max_num_samples - num_valid_labels

            if num_padding > 0:
                # 从全局负样本池中随机采样
                global_neg_len = self.global_grounding_neg_embeddings.shape[0]

                # 随机选择索引
                pad_indices = np.random.choice(
                    global_neg_len,
                    size=num_padding,
                    replace=(num_padding > global_neg_len)  # 如果需要填补的比池子还大，就允许重复
                )

                # 获取对应的 Embeddings
                pad_embeddings = self.global_grounding_neg_embeddings[pad_indices]

                # 拼接到结果后面
                txt_feats = torch.cat((txt_feats, pad_embeddings), dim=0)

        # 最终将 Tensor 赋值给 results['texts']
        # 注意：后续的 Pipeline (如 PackDetInputs) 需要能够处理 texts 是 Tensor 的情况
        # 或者你在 PackDetInputs 里把这个 Tensor 放到 data_samples.texts 里面
        results['texts'] = txt_feats

        return results


@TRANSFORMS.register_module()
class LoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        texts = []
        for idx, cls_caps in enumerate(class_texts):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        results['texts'] = texts

        return results


@TRANSFORMS.register_module()
class LoadVisualMask:
    def __init__(self, scale_factor=1/8, min_iterval=5):
        self.scale_factor = scale_factor
        self.min_iterval = min_iterval

    def make_mask(self, boxes, h, w):
        """
                生成 Mask 的核心逻辑
                boxes: shape (N, 4), 格式为 xyxy (绝对坐标)
                h, w: mask 的高和宽
        """
        #确保 bboxes 至少为2d
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

        #这里的bboxes已经是缩放后的xyxy格式
        x1 = boxes[:, 0].reshape(-1, 1, 1)
        y1 = boxes[:, 1].reshape(-1, 1, 1)
        x2 = boxes[:, 2].reshape(-1, 1, 1)
        y2 = boxes[:, 3].reshape(-1, 1, 1)

        #生成网络
        r = torch.arange(w, device=boxes.device)[None, None, :]
        c = torch.arange(h, device=boxes.device)[None, :, None]

        return ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def __call__(self, results):
        #1.获取图像尺寸（640，640）
        img_h, img_w = results['img_shape'][:2]

        #2.计算mask尺寸
        mask_h, mask_w = int(img_h * self.scale_factor), int(img_w * self.scale_factor)

        #3.获取并处理bboding bboxes
        if 'gt_bboxes' in results:
            gt_bboxes_obj = results['gt_bboxes']
            #注意：results['gt_bboxes'] 是 HorizontalBoxes 对象，需要取 .tensor
            bboxes = gt_bboxes_obj.tensor

            #缩放bboxes到mask尺寸
            scale_x = mask_w / img_w
            scale_y = mask_h / img_h

            # 创建缩放后的副本，避免修改原始 gt_bboxes
            scaled_bboxes = bboxes.clone()
            scaled_bboxes[:, 0] *= scale_x
            scaled_bboxes[:, 1] *= scale_y
            scaled_bboxes[:, 2] *= scale_x
            scaled_bboxes[:, 3] *= scale_y

            masks = self.make_mask(scaled_bboxes, mask_h, mask_w).float()
        else:
            raise ValueError("LoadVisualMask requires 'gt_bboxes' in results")

        labels = results['gt_bboxes_labels']
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
        else:
            labels = labels.long()

        # 这里的逻辑是：为每个出现的类别创建一个 mask 通道
        # 如果是 Open Vocabulary，通常类别数是当前 batch 的文本数量或 max_label + 1
        if len(labels) > 0:
            num_classes = labels.max().item() + 1
            visuals = torch.zeros(num_classes, mask_h, mask_w, device=masks.device)

            for idx, mask in zip(labels, masks):
                visuals[idx] = torch.logical_or(visuals[idx].bool(), mask.bool()).float()
        else:
            visuals = torch.zeros(0, mask_h, mask_w)

        results['visual_masks'] = visuals

        return results








