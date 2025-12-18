# import torch
# import torch.nn.functional as F
# from mmengine.hooks import Hook
# from mmengine.runner import Runner
# from mmyolo.registry import HOOKS
# from mmengine.model import is_model_wrapper
# from tqdm import tqdm
#
#
# @HOOKS.register_module()
# class VisualPromptInjectionHook(Hook):
#     def __init__(self, dataloader_cfg, num_classes=1203):
#         """
#         Args:
#             dataloader_cfg (dict): 验证用 VPS 数据集的配置
#             num_classes (int): LVIS 类别数
#         """
#         self.dataloader_cfg = dataloader_cfg
#         self.num_classes = num_classes
#         self.vps_loader = None
#
#     def _inject_visual_prompts(self, runner: Runner):
#         """
#         在验证 epoch 开始前，计算 Visual Prototypes 并注入模型。
#         """
#         # 1. 构建 DataLoader (利用 Runner 的机制，只构建一次)
#         if self.vps_loader is None:
#             self.vps_loader = runner.build_dataloader(self.dataloader_cfg)
#
#         model = runner.model
#         if is_model_wrapper(model):
#             model = model.module
#
#         device = next(model.parameters()).device
#
#         # 假设 bbox_head 中有 savpe 模块或通过其他方式获取 embed_dim
#         # 如果 savpe 在 bbox_head 中，通常维度是 model.bbox_head.head_module.embed_dims
#         embed_dim = 512  # 请根据实际情况修改，或动态获取 model.bbox_head.embed_dims
#
#         # 2. 初始化累加器 (Sum) 和 计数器 (Count)
#         visual_sum = torch.zeros(self.num_classes, embed_dim, device=device)
#         cls_counts = torch.zeros(self.num_classes, device=device)
#
#         runner.logger.info(f"[VisualPromptHook] Generating prototypes from {len(self.vps_loader.dataset)} samples...")
#
#         model.eval()
#
#         # 使用 tqdm 显示进度条
#         pbar = tqdm(self.vps_loader, desc="Extracting Visual Prompts")
#
#         with torch.no_grad():
#             for data in pbar:
#                 # 预处理数据 (移动到 GPU, 归一化等)
#                 data = model.data_preprocessor(data, training=False)
#
#                 inputs = data['inputs']
#                 data_samples = data['data_samples']
#
#                 # A. 提取基础特征 (Backbone + Neck)
#                 # 这对应 model.extract_feat(img)
#                 #feats = model.extract_feat(inputs)
#                 img_feats, txt_feats, txt_masks = model.extract_feat(inputs, data_samples)
#
#                 # B. 遍历 Batch 中的每张图片
#                 for j, sample in enumerate(data_samples):
#                     # 获取该样本的类别 (YOLOE 逻辑：每张 Crop 图片只对应一个类别)
#                     gt_labels = sample.gt_instances.labels
#                     if len(gt_labels) == 0:
#                         continue
#
#                     # 取出类别 ID
#                     target_cls = gt_labels[0].long()
#
#                     # 获取 Visual Masks (假设在 data pipeline 中已经加载并放入 data_samples)
#                     # 形状应该处理为 (1, H, W) 或 (H, W)
#                     if hasattr(sample, 'visual_masks'):
#                         v_mask = sample.visual_masks
#                     elif hasattr(sample, 'metainfo') and 'visual_masks' in sample.metainfo:
#                         v_mask = sample.metainfo['visual_masks']
#                     else:
#                         continue
#
#                     # 确保 mask 在 GPU 上且维度正确
#                     # SAVPE 输入要求: visual_masks (B, Q, H, W)
#                     # 这里 Batch=1, Q=1 -> (1, 1, H, W)
#                     if v_mask.ndim == 2:
#                         v_mask = v_mask.unsqueeze(0).unsqueeze(0)
#                     elif v_mask.ndim == 3:
#                         v_mask = v_mask.unsqueeze(0)
#
#                     v_mask = v_mask.to(device)
#
#                     # 切片取出当前图片的特征 (List of tensors)
#                     # feats 是多尺度的 list, 每个元素 shape (B, C, H, W) -> 切片为 (1, C, H, W)
#                     single_img_feats = [f[j:j + 1] for f in img_feats]
#
#                     # C. 调用 SAVPE 提取 Embedding
#                     # 对应 YOLOE: model.get_visual_pe -> savpe(feats, mask)
#                     # 假设 savpe 位于 model.bbox_head.savpe
#                     # 输出 shape: (1, 1, 512)
#                     prompt_embeds = model.bbox_head.head_module.savpe(single_img_feats, v_mask)
#
#                     # D. 累加特征
#                     # squeeze -> (512,)
#                     if prompt_embeds.shape[1] > 0:
#                         visual_sum[target_cls] += prompt_embeds.squeeze()
#                         cls_counts[target_cls] += 1
#
#         # 3. 计算平均值 (Mean)
#         # 避免除以 0
#         mask = cls_counts > 0
#         # visual_pe: (Num_Classes, 512)
#         visual_pe = torch.zeros_like(visual_sum)
#         visual_pe[mask] = visual_sum[mask] / cls_counts[mask].unsqueeze(-1)
#
#         # 4. L2 归一化 (对应 YOLOE 的 F.normalize)
#         visual_pe[mask] = F.normalize(visual_pe[mask], dim=-1, p=2)
#
#         # 5. 调整形状并注入 (Injection)
#         # YOLOE 输出是 (1, NC, Dim)，这里我们也保持一致
#         # 注入到 Head 中
#         final_pe = visual_pe.unsqueeze(0)  # (1, 1203, 512)
#
#         # 调用 Head 的方法进行注入
#         if hasattr(model.bbox_head, 'set_visual_prototypes'):
#             model.bbox_head.set_visual_prototypes(final_pe)
#             runner.logger.info(f"Successfully injected visual prompts. Shape: {final_pe.shape}")
#         else:
#             runner.logger.warning("Warning: bbox_head does not have 'set_visual_prototypes' method!")
#
#     def before_val_epoch(self, runner: Runner):
#         """训练中的验证阶段调用"""
#         self._inject_visual_prompts(runner)
#
#     def before_test_epoch(self, runner: Runner):
#         """独立测试脚本 (tools/test.py) 调用"""
#         self._inject_visual_prompts(runner)


import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.runner import Runner

try:
    from mmyolo.registry import HOOKS
except ImportError:
    from mmdet.registry import HOOKS
from mmengine.model import is_model_wrapper
from tqdm import tqdm


@HOOKS.register_module()
class VisualPromptInjectionHook(Hook):
    def __init__(self, dataloader_cfg, num_classes=1203):
        """
        Args:
            dataloader_cfg (dict): 验证用 VPS 数据集的配置
            num_classes (int): LVIS 类别数
        """
        self.dataloader_cfg = dataloader_cfg
        self.num_classes = num_classes
        self.vps_loader = None
        self._has_injected = False

    def _inject_visual_prompts(self, runner: Runner):
        """
        在验证 epoch 开始前，计算 Visual Prototypes 并注入模型。
        """
        # 1. 构建 DataLoader
        if self.vps_loader is None:
            self.vps_loader = runner.build_dataloader(self.dataloader_cfg)

        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(model.parameters()).device

        # 获取 embed_dim
        if hasattr(model.bbox_head, 'head_module'):
            embed_dim = model.bbox_head.head_module.embed_dims
        else:
            embed_dim = 512

            # 初始化累加器
        visual_sum = torch.zeros(self.num_classes, embed_dim, device=device)
        cls_counts = torch.zeros(self.num_classes, device=device)

        runner.logger.info(f"[VisualPromptHook] Generating prototypes from {len(self.vps_loader.dataset)} samples...")

        model.eval()
        pbar = tqdm(self.vps_loader, desc="Extracting Visual Prompts")

        with torch.no_grad():
            for data in pbar:
                data = model.data_preprocessor(data, training=False)
                inputs = data['inputs']
                data_samples = data['data_samples']

                img_feats, txt_feats, txt_masks = model.extract_feat(inputs, data_samples)

                batch_masks_list = []
                batch_labels_list = []
                valid_indices = []

                for i, sample in enumerate(data_samples):
                    gt_labels = sample.gt_instances.labels
                    if len(gt_labels) == 0:
                        continue

                    target_cls = gt_labels[0].long()

                    # 获取原始 Mask (C, H, W) - C 取决于类别ID
                    if hasattr(sample, 'visual_masks'):
                        v_mask = sample.visual_masks
                    elif hasattr(sample, 'metainfo') and 'visual_masks' in sample.metainfo:
                        v_mask = sample.metainfo['visual_masks']
                    else:
                        continue

                    # [关键修改]
                    # YOLOE 的 LoadVisualMask 生成的 mask 是 (LabelID+1, H, W)
                    # 我们只取当前类别对应的那个通道，使其变为 (H, W)
                    # 这样所有样本的 shape 就一致了，可以 stack

                    # 边界检查，防止 mask 维度不够 (比如 Transform 没写对)
                    if v_mask.shape[0] <= target_cls:
                        # 这种情况理论不该发生，除非 LoadVisualMask 逻辑变了
                        continue

                    # 提取特定通道: (H, W)
                    specific_mask = v_mask[target_cls]

                    # 扩展为 (1, H, W)
                    specific_mask = specific_mask.unsqueeze(0)

                    batch_masks_list.append(specific_mask)
                    batch_labels_list.append(target_cls)
                    valid_indices.append(i)

                if len(batch_masks_list) == 0:
                    continue

                # 堆叠: (B_valid, 1, H, W)
                batch_masks = torch.stack(batch_masks_list).to(device)
                batch_labels = torch.stack(batch_labels_list).to(device)

                # 筛选特征
                if len(valid_indices) == inputs.shape[0]:
                    curr_img_feats = img_feats
                else:
                    curr_img_feats = [feat[valid_indices] for feat in img_feats]

                # 调用 SAVPE
                # 输入: mask shape (B, 1, H, W)
                # 输出: shape (B, 1, 512)
                # 因为我们只传了 1 个通道的 mask，所以 output 的 dim=1 也是 1
                batch_prompt_embeds = model.bbox_head.head_module.savpe(curr_img_feats, batch_masks)

                # 变成 (B, 512)
                batch_prompt_embeds = batch_prompt_embeds.squeeze(1)

                # 累加
                visual_sum.index_add_(0, batch_labels, batch_prompt_embeds)

                # 计数
                ones = torch.ones_like(batch_labels, dtype=torch.float)
                cls_counts.index_add_(0, batch_labels, ones)
        """
                cls_counts是长为1203的一维张量，只有后面两个维度为0
                visuall_sum是shape为[1203,512]的二维张量，后面两行对应的512维全为0
        """

        #设定维度常量
        # N = 1203
        # DIM = 512
        # counts_part1 = torch.randint(low=1, high=100, size=(N - 2,))
        # counts_part2 = torch.zeros(2, dtype=torch.long)
        # cls_counts = torch.cat([counts_part1, counts_part2])
        #
        # visual_sum = torch.randn(N, DIM)
        # visual_sum[-2:, :] = 0.0
        # cls_counts = cls_counts.to(device)
        # visual_sum = visual_sum.to(device)

        # 后处理：平均 + 归一化
        mask = cls_counts > 0
        visual_pe = torch.zeros_like(visual_sum)
        visual_pe[mask] = visual_sum[mask] / cls_counts[mask].unsqueeze(-1)
        visual_pe[mask] = F.normalize(visual_pe[mask], dim=-1, p=2)

        # 注入
        final_pe = visual_pe.unsqueeze(0)
        if hasattr(model.bbox_head, 'set_visual_prototypes'):
            model.bbox_head.set_visual_prototypes(final_pe)
            runner.logger.info(f"Successfully injected visual prompts. Shape: {final_pe.shape}")
        else:
            runner.logger.warning("Warning: bbox_head does not have 'set_visual_prototypes' method!")

    def before_val_epoch(self, runner: Runner):
        """训练中的验证阶段调用"""
        self._inject_visual_prompts(runner)

    def before_test_epoch(self, runner: Runner):
        """独立测试脚本 (tools/test.py) 调用"""
        self._inject_visual_prompts(runner)