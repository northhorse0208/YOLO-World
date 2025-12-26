# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model.weight_init import constant_init, xavier_init




@MODELS.register_module()
class ContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """
    def __init__(self,
                 embed_dims: int,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class BNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """
    def __init__(self,
                 embed_dims: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class RepBNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """
    def __init__(self,
                 embed_dims: int,
                 num_guide_embeds: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.conv = nn.Conv2d(embed_dims, num_guide_embeds, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        x = self.conv(x)
        return x


@MODELS.register_module()
class YOLOWorldHeadModule(YOLOv8HeadModule):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
    """
    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()
        #self.savpe = SAVPE(in_channels=[256, 512, 512], hidden_channels=256, embed_dims=512)
        self.savpe = DeformablePromptEncoder()
        self.opr_fusion = OrthogonalFusionModule(embed_dims=self.embed_dims)

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum))
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    # def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
    #             txt_masks: Tensor, visual_masks: Tensor, bboxes_labels: Tensor) -> Tuple[List]:
    #     """Forward features from the upstream network."""
    #     assert len(img_feats) == self.num_levels
    #     if visual_masks.shape[-1] == 80:
    #         visual_embeds = self.savpe(img_feats, visual_masks)
    #     elif visual_masks.shape[-1] == 512:
    #         visual_embeds = visual_masks
    #     else:
    #         raise ValueError("shape cuo wu")
    #     txt_feats = visual_embeds
    #     txt_feats = [txt_feats for _ in range(self.num_levels)]
    #     txt_masks = [txt_masks for _ in range(self.num_levels)]
    #     return multi_apply(self.forward_single, img_feats, txt_feats,
    #                        txt_masks, self.cls_preds, self.reg_preds,
    #                        self.cls_contrasts)

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
                txt_masks: Tensor, visual_masks: Tensor,
                bboxes_labels: Tensor) -> Tuple[List]:
        """
        Forward features with Orthogonal Fusion guided by Bbox Labels.
        Args:
            img_feats: FPN Features
            txt_feats: (B, 80, 512)
            txt_masks: 用于 loss
            visual_masks: (B, N, H, W)
            bboxes_labels: (M, 6) -> [batch_idx, cls_id, x1, y1, x2, y2]
                           这是 batch 中所有图片的 GT 拼接在一起的张量
        """
        assert len(img_feats) == self.num_levels

        # 1. 提取视觉 Embeddings
        # visual_embeds: (B, N, 512)
        if visual_masks.shape[-1] == 80:
            visual_embeds = self.savpe(img_feats, visual_masks)
        elif visual_masks.shape[-1] == 512:
            visual_embeds = visual_masks
        else:
            raise ValueError('shape fault')

        # 2. 交互与融合
        fused_txt_feats = txt_feats.clone()
        B = txt_feats.shape[0]
        align_loss = torch.tensor(0.0, device=txt_feats.device)
        num_valid_samples = 0  # 用于平均 loss

        # 解析 bboxes_labels 以获取每张图的正类别数量 K
        # bboxes_labels[:, 0] 是 batch index
        # bboxes_labels[:, 1] 是 class id

        batch_indices = bboxes_labels[:, 0].long()
        class_ids = bboxes_labels[:, 1].long()

        for i in range(B):
            # 找出当前图片 (batch_idx == i) 的所有 class_ids
            current_img_mask = (batch_indices == i)

            if not current_img_mask.any():
                # 如果这张图没有 GT (也就是全是负样本)，直接跳过融合
                continue

            current_classes = class_ids[current_img_mask]

            # 去重并排序，因为 text_feats 和 visual_embeds 应该是按类别顺序排列的唯一集合
            # unique() 会自动排序，这很重要！确保和 Pipeline 中的逻辑一致
            unique_classes = torch.unique(current_classes)

            # K 就是当前图的正类别数量
            K = len(unique_classes)

            # 理论上 K 应该等于 visual_embeds 的有效长度，也等于 txt_feats 中前 K 个正样本
            # 做一个防御性编程，防止 K 超过了 txt_feats 的总容量 (80)
            K = min(K, txt_feats.shape[1])

            if K == 0:
                continue

            # # 取出对应的特征
            # vis_k = visual_embeds[i, :K, :]
            # txt_k = txt_feats[i, :K, :]
            # # [新增] 计算对齐损失: 1 - cosine_similarity
            # # dim=-1 计算向量间的相似度
            # sim = F.cosine_similarity(vis_k, txt_k, dim=-1)  # (K,)
            # # loss = 1 - sim (范围 0~2, 越小越好)
            # align_loss += (1.0 - sim).sum()
            # num_valid_samples += K
            #
            # # --- OPR 融合 ---
            # fused_k = self.opr_fusion(txt_k, vis_k)
            #
            # # 填回
            # fused_txt_feats[i, :K, :] = fused_k

            # 取出对应的特征
            vis_k = visual_embeds[i, :K, :]
            txt_k = txt_feats[i, :K, :]

            # --- OPR 融合 ---
            # 注意：这里接收两个返回值
            fused_k, v_proj_k = self.opr_fusion(txt_k, vis_k)

            # [修正] 计算对齐损失
            # 现在我们计算的是 "映射后的视觉特征" 与 "文本特征" 的距离
            # 这才真正约束了 Adapter 的学习方向
            sim = F.cosine_similarity(v_proj_k, txt_k, dim=-1)
            align_loss += (1.0 - sim).sum()
            num_valid_samples += K

            # 填回
            fused_txt_feats[i, :K, :] = fused_k

            # stage1 只训练deformablepromptencoder，但是也加入对齐损失
            # 1. 投影映射 (Projection)
            # 我们直接调用 OPR 模块里的 visual_adapter 子模块
            # 这样既复用了代码，又保证了权重位置正确
            # v_proj_k = self.opr_fusion.visual_adapter(vis_k)  # (K, 512)
            #
            # # 2. 计算对齐损失 (Alignment Loss)
            # # 强制投影后的视觉特征 v_proj_k 靠近 CLIP 文本特征 txt_k
            # sim = F.cosine_similarity(v_proj_k, txt_k, dim=-1)
            # align_loss += (1.0 - sim).sum()
            # num_valid_samples += K
            #
            # # 3. 特征替换 (Replacement)
            # # 我们完全跳过 opr_fusion.forward 的融合逻辑
            # # 直接把 v_proj_k 填入 fused_txt_feats
            # # 此时 Head 接收到的就是 "伪装成文本的视觉特征"
            # fused_txt_feats[i, :K, :] = v_proj_k

            # 平均 Loss
        if num_valid_samples > 0:
            align_loss = align_loss / num_valid_samples
        # 3. 后续处理
        txt_feats_list = [fused_txt_feats for _ in range(self.num_levels)]
        txt_masks_list = [txt_masks for _ in range(self.num_levels)]

        return multi_apply(self.forward_single, img_feats, txt_feats_list,
                           txt_masks_list, self.cls_preds, self.reg_preds,
                           self.cls_contrasts), align_loss

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       txt_masks: Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)

        if txt_masks is not None:
            txt_masks = txt_masks.view(b, -1, 1, 1).expand(-1, -1, h, w)
            if self.training:
                cls_logit = cls_logit * txt_masks
                cls_logit[txt_masks == 0] = -10e6
            else:
                cls_logit[txt_masks == 0] = -10e6

        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class RepYOLOWorldHeadModule(YOLOWorldHeadModule):
    def __init__(self,
                 *args,
                 embed_dims: int,
                 num_guide: int,
                 freeze_all: bool = False,
                 **kwargs) -> None:
        super().__init__(*args,
                         embed_dims=embed_dims,
                         use_bn_head=True,
                         use_einsum=False,
                         freeze_all=freeze_all,
                         **kwargs)

        # using rep head
        cls_contrasts = []
        for _ in range(self.num_levels):
            cls_contrasts.append(
                RepBNContrastiveHead(embed_dims=embed_dims,
                                     num_guide_embeds=num_guide,
                                     norm_cfg=self.norm_cfg))
        self.cls_contrasts = nn.ModuleList(cls_contrasts)

    def forward_single(self, img_feat: Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward features from the upstream network."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds

    def forward(self, img_feats: Tuple[Tensor]) -> Tuple[List]:
        assert len(img_feats) == self.num_levels
        return multi_apply(self.forward_single, img_feats, self.cls_preds,
                           self.reg_preds, self.cls_contrasts)


@MODELS.register_module()
class YOLOWorldHead(YOLOv8Head):
    """YOLO-World Head
    """
    def __init__(self, world_size=-1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.world_size = world_size

    """YOLO World v8 head."""

    def set_visual_prototypes(self, embeddings):
        """Set custom prototypes (e.g. visual embeddings from LVIS) for inference."""
        self.custom_prototypes = embeddings.detach()

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             txt_masks: Tensor, batch_data_samples: Union[list, dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""

        #outs = self(img_feats, txt_feats, txt_masks)
        outs, align_loss = self(img_feats, txt_feats, txt_masks, batch_data_samples['visual_masks'], batch_data_samples['bboxes_labels'])
        # Fast version
        loss_inputs = outs + (txt_masks, batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)
        losses['loss_align'] = align_loss * 1.0

        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        txt_masks: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats, txt_masks)

        loss_inputs = outs + (txt_masks, batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
                txt_masks: Tensor, visual_masks: Tensor, bbox_labels: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats, txt_feats, txt_masks, visual_masks, bbox_labels)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                txt_masks: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        
        col_0 = torch.zeros(1203)
        col_1 = torch.arange(1203)
        val_imitate_bbox_labels = torch.stack((col_0, col_1), dim=1)
        val_imitate_bbox_labels = val_imitate_bbox_labels.to('cuda')
        
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs, _ = self(img_feats, txt_feats, txt_masks, self.custom_prototypes, val_imitate_bbox_labels)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_text_masks: Tensor,
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        num_curr_classes = cls_scores[0].shape[1]
        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 num_curr_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        if hasattr(self.assigner, 'num_classes'):
            self.assigner.num_classes = num_curr_classes
        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        if batch_text_masks is not None:
            cls_weight = batch_text_masks.view(num_imgs, 1, -1).expand(
                -1, flatten_cls_preds.shape[1], -1).to(flatten_cls_preds)

            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
            _loss_cls = (loss_cls * cls_weight).sum(dim=-1)
            loss_cls = _loss_cls.sum()
        else:
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]
        # 8400
        # print(flatten_cls_scores.shape)
        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list


class SAVPE(BaseModule):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 embed_dims,
                 c_internal=16,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='SiLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.c = c_internal

        conv_cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.cv1 = nn.ModuleList()
        for ch in in_channels:
            layers = [
                ConvModule(ch, hidden_channels, 3, padding=1, **conv_cfg),
                ConvModule(hidden_channels, hidden_channels, 3, padding=1, **conv_cfg)
            ]
            self.cv1.append(nn.Sequential(*layers))

        # 为 P4 (index 1) 和 P5 (index 2) 添加上采样，使其与 P3 对齐
        self.cv1[1].add_module('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
        self.cv1[2].add_module('upsample', nn.Upsample(scale_factor=4, mode='nearest'))

        # 2. 构建 cv2 分支 (用于提取 Shallow/Spatial Features)
        # 原逻辑: [Conv(x, c3, 1)] + Upsample
        self.cv2 = nn.ModuleList()
        for ch in in_channels:
            layers = [
                ConvModule(ch, hidden_channels, 1, **conv_cfg)  # 1x1 Conv
            ]
            self.cv2.append(nn.Sequential(*layers))

        self.cv2[1].add_module('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
        self.cv2[2].add_module('upsample', nn.Upsample(scale_factor=4, mode='nearest'))

        # 3. 聚合层
        # 对应: self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        # 注意：原代码 cv3, cv4, cv5 是纯卷积还是带激活的？
        # 根据 Ultralytics 习惯，单独写 nn.Conv2d 通常是不带 BN/Act 的，
        # 但如果是 Feature Projection，通常用 ConvModule (1x1) 更好。
        # 这里为了严格还原逻辑，cv3/4/5 使用纯 Conv2d，除非原代码中引入了 Conv 块。
        # 既然原代码写的是 nn.Conv2d，我们就用 nn.Conv2d。

        self.cv3 = nn.Conv2d(len(in_channels) * hidden_channels, embed_dims, 1)
        self.cv4 = nn.Conv2d(len(in_channels) * hidden_channels, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)  # 处理 mask 的卷积

        # 4. 融合层 cv6
        # 原逻辑: Sequential(Conv(2*c, c, 3), nn.Conv2d(c, c, 3, padding=1))
        # 第一层用 ConvModule (带BN/SiLU)，第二层用纯 Conv2d
        self.cv6 = nn.Sequential(
            ConvModule(2 * self.c, self.c, 3, padding=1, **conv_cfg),
            nn.Conv2d(self.c, self.c, 3, padding=1)
        )

    def forward(self, x, vp):
        """
        Args:
            x (list[Tensor]): 多尺度特征图 [P3, P4, P5]
            vp (Tensor): Visual Prompts / Masks, shape (B, Q, H, W)
                         这里 H, W 应该对应 P3 的分辨率 (即 1/8 原图)
        """
        # --- 分支 2 处理 (Query生成相关) ---
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        # 拼接并在通道维度映射
        y = self.cv4(torch.cat(y, dim=1))

        # --- 分支 1 处理 (Value/Key生成相关) ---
        x_feats = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x_feats = self.cv3(torch.cat(x_feats, dim=1))

        B, C, H, W = x_feats.shape

        # x: (B, C, H*W) -> (B, C, N)
        x_flat = x_feats.view(B, C, -1)

        # 获取 Prompt 数量 Q
        Q = vp.shape[1]

        # --- 复杂的 Reshape 和 Expand 逻辑 ---
        # 这里的目的是将 Visual Prompt (Q个) 与 图像特征进行融合

        # y 扩展为 (B*Q, c, H, W)
        # y: (B, c, H, W) -> (B, 1, c, H, W) -> (B, Q, c, H, W) -> (B*Q, c, H, W)
        y_expanded = y.unsqueeze(1).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)

        # vp 扩展为 (B*Q, 1, H, W)
        vp_expanded = vp.reshape(B * Q, 1, H, W)

        # 特征融合: 图像特征 y 与 mask 特征 vp_expanded 拼接
        # (B*Q, 2*c, H, W) -> (B*Q, c, H, W)
        y_fused = self.cv6(torch.cat((y_expanded, self.cv5(vp_expanded)), dim=1))

        # 恢复维度为 (B, Q, c, H*W)
        y_fused = y_fused.reshape(B, Q, self.c, -1)

        # mask 恢复维度 (B, Q, 1, H*W)
        vp_flat = vp.reshape(B, Q, 1, -1)

        # --- Attention / Masking 机制 ---
        # 如果 vp 是 0 (背景)，则加上极小值，使 Softmax 后为 0
        # 这一步是为了只关注 mask 区域内的特征
        score = y_fused * vp_flat + torch.logical_not(vp_flat) * torch.finfo(y_fused.dtype).min

        # Softmax over spatial dimensions (last dim)
        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        # --- 聚合特征 ---
        # score: (B, Q, c, N) -> transpose -> (B, Q, N, c)
        # x_flat: (B, C, N) -> reshape -> (B, c, C/c, N) -> transpose -> (B, c, N, C/c)
        # Matmul: (B, Q, N, c) @ (B, c, N, C/c) ??? 维度好像对不上

        # 让我们仔细看原代码的矩阵乘法：
        # aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        # score shape: (B, Q, c, N)
        # score.transpose(-2, -3) -> (B, c, Q, N)  <-- 注意这里

        # x shape: (B, C, N)
        # x_reshaped: (B, c, C//c, N)
        # x_reshaped.transpose(-1, -2): (B, c, N, C//c)

        # Matmul: (B, c, Q, N) @ (B, c, N, C//c)
        # Result: (B, c, Q, C//c)

        # Re-implementing strictly:
        score_t = score.transpose(1, 2)  # (B, c, Q, N)

        x_reshaped_t = x_flat.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)  # (B, c, N, C//c)

        aggregated = score_t @ x_reshaped_t  # (B, c, Q, C//c)

        # transpose(-2, -3) -> (B, Q, c, C//c)
        # reshape -> (B, Q, C)
        final_out = aggregated.transpose(1, 2).reshape(B, Q, -1)

        return F.normalize(final_out, dim=-1, p=2)


# class DeformablePromptEncoder(BaseModule):
#     def __init__(self,
#                  in_channels=[256, 512, 512],  # [256, 512, 1024]
#                  hidden_channels=256,  # 256 (用于中间投影)
#                  embed_dims=512,  # 512 (最终输出维度，需与 text embedding 一致)
#                  num_heads=8,
#                  num_points=4,  # 每个 head 采样的点数
#                  feedforward_channels=1024,
#                  dropout=0.1,
#                  norm_cfg=dict(type='GN', num_groups=32),
#                  init_cfg=None):
#         super().__init__(init_cfg)
#
#         self.embed_dims = embed_dims
#
#         # 1. 多尺度特征投影层 (将 P3, P4, P5 统一映射到 embed_dims)
#         self.input_projs = nn.ModuleList()
#         for ch in in_channels:
#             self.input_projs.append(
#                 nn.Sequential(
#                     nn.Conv2d(ch, embed_dims, 1),
#                     build_norm_layer(norm_cfg, embed_dims)[1]
#                 )
#             )
#
#         # 2. Level Embeddings (区分不同尺度)
#         self.level_embeds = nn.Parameter(torch.Tensor(len(in_channels), embed_dims))
#
#         # 3. Deformable Attention 核心层
#         # batch_first=True: 输入 query 为 (B, N, C)
#         self.cross_attn = MultiScaleDeformableAttention(
#             embed_dims=embed_dims,
#             num_levels=len(in_channels),
#             num_heads=num_heads,
#             num_points=num_points,
#             dropout=dropout,
#             batch_first=True
#         )
#
#         # 4. FFN (Feed-Forward Network)
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dims, feedforward_channels),
#             nn.GELU(),
#             nn.Linear(feedforward_channels, embed_dims)
#         )
#
#         # 5. Norms
#         self.norm1 = nn.LayerNorm(embed_dims)
#         self.norm2 = nn.LayerNorm(embed_dims)
#
#         # 6. Query 初始化投影 (将 P3 特征用于初始化 Query)
#         # SAVPE 输入的是 list，我们通常取分辨率最高的 P3 做 mask pooling
#         self.query_init_proj = nn.Conv2d(in_channels[0], embed_dims, 1)
#
#     def init_weights(self):
#         super().init_weights()
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, MultiScaleDeformableAttention):
#                 m.init_weights()
#         xavier_init(self.level_embeds, distribution='uniform')
#
#     def get_reference_points(self, visual_masks):
#         """
#         计算 Mask 重心作为 Deformable Attention 的参考点。
#         Args:
#             visual_masks: (B, N, H, W) - H, W 对应 P3 分辨率
#         Returns:
#             reference_points: (B, N, 1, 2) 归一化坐标 (cx, cy)
#         """
#         B, N, H, W = visual_masks.shape
#         device = visual_masks.device
#
#         # 生成网格 (y, x)
#         y_grid, x_grid = torch.meshgrid(
#             torch.arange(H, device=device, dtype=torch.float32),
#             torch.arange(W, device=device, dtype=torch.float32),
#             indexing='ij'
#         )
#
#         # 增加维度以广播: (1, 1, H, W)
#         y_grid = y_grid.unsqueeze(0).unsqueeze(0)
#         x_grid = x_grid.unsqueeze(0).unsqueeze(0)
#
#         # 加个 epsilon 防止除零 (有些 mask 可能是空的)
#         mask_sum = visual_masks.sum(dim=[-1, -2]) + 1e-6
#
#         # 计算重心: sum(coord * weight) / sum(weight)
#         # visual_masks 是 0/1 或者是 float 的权重
#         cx = (visual_masks * x_grid).sum(dim=[-1, -2]) / mask_sum
#         cy = (visual_masks * y_grid).sum(dim=[-1, -2]) / mask_sum
#
#         # 归一化到 [0, 1]
#         # 注意: x 对应 W, y 对应 H
#         cx_norm = cx / W
#         cy_norm = cy / H
#
#         # 堆叠 -> (B, N, 2) -> (B, N, 1, 2) (1 代表 num_levels 维度广播)
#         ref_points = torch.stack([cx_norm, cy_norm], dim=-1).unsqueeze(2)
#
#         # 截断到有效范围，防止 NaN 或越界
#         return ref_points.clamp(0.01, 0.99)
#
#     def forward(self, x, vp):
#         """
#         Args:
#             x (list[Tensor]): [P3, P4, P5]
#             vp (Tensor): Visual Masks (B, N, H, W) - 对应 P3 分辨率
#         Returns:
#             out (Tensor): (B, N, embed_dims)
#         """
#         # x[0] 是 P3, 形状 (B, C0, H, W)
#         B, N, H, W = vp.shape
#
#         # -----------------------------------------------------------
#         # 1. 准备 Multi-scale Key/Value
#         # -----------------------------------------------------------
#         src_flattens = []
#         spatial_shapes = []
#
#         for idx, feat in enumerate(x):
#             # 投影到 embed_dims
#             src = self.input_projs[idx](feat)  # (B, C, H_i, W_i)
#             bs, c, h_i, w_i = src.shape
#             spatial_shapes.append((h_i, w_i))
#
#             # Flatten: (B, H_i*W_i, C)
#             src = src.flatten(2).transpose(1, 2)
#
#             # 加上 Level Embedding (区分 P3, P4, P5)
#             src = src + self.level_embeds[idx].view(1, 1, -1)
#             src_flattens.append(src)
#
#         # 拼接所有层特征: (B, Sum_HW, C)
#         src_flattens = torch.cat(src_flattens, 1)
#
#         # 构建 MSDA 需要的 spatial_shapes 和 level_start_index
#         spatial_shapes = torch.tensor(spatial_shapes, device=vp.device, dtype=torch.long)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#
#         # -----------------------------------------------------------
#         # 2. 初始化 Query (Content-Aware Initialization)
#         # -----------------------------------------------------------
#         # 使用 Mask 在 P3 特征上做加权平均，作为 Query 的初始值
#         # 相比 SAVPE，我们这里只用 P3 做初始化，依靠 Attention 去看 P4/P5
#
#         p3_feat = x[0]  # (B, C_in, H, W)
#         p3_proj = self.query_init_proj(p3_feat)  # (B, C_embed, H, W)
#
#         # Mask Pooling
#         # vp: (B, N, H, W) -> (B, N, 1, H, W)
#         vp_expanded = vp.unsqueeze(2)
#         # p3: (B, C, H, W) -> (B, 1, C, H, W)
#         p3_expanded = p3_proj.unsqueeze(1)
#
#         mask_sum = vp_expanded.sum(dim=[-1, -2]) + 1e-6
#         # Sum Pooling -> (B, N, C)
#         query_init = (p3_expanded * vp_expanded).sum(dim=[-1, -2]) / mask_sum
#
#         # -----------------------------------------------------------
#         # 3. 准备 Reference Points
#         # -----------------------------------------------------------
#         reference_points = self.get_reference_points(vp)  # (B, N, 1, 2)
#
#         # -----------------------------------------------------------
#         # 4. Deformable Attention Cross-Interaction
#         # -----------------------------------------------------------
#         # query: (B, N, C)
#         # value: (B, Sum_HW, C)
#         query_interact = self.cross_attn(
#             query=query_init,
#             value=src_flattens,
#             reference_points=reference_points,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index
#         )
#
#         # Residual + Norm
#         query_out = self.norm1(query_init + query_interact)
#
#         # -----------------------------------------------------------
#         # 5. FFN
#         # -----------------------------------------------------------
#         query_final = self.norm2(query_out + self.ffn(query_out))
#
#         # Output: (B, N, 512)
#         # 这里的 N 就是 mask 的通道数，不需要 reshape 操作
#         # 且我们使用了 L2 Norm 在外部 Hook 做，这里保持 raw feature 即可
#         # 如果需要保持和 SAVPE 完全一致的输出分布，可以在这里加 F.normalize
#         return F.normalize(query_final, dim=-1, p=2)

class DeformablePromptEncoder(BaseModule):
    def __init__(self,
                 in_channels=[256, 512, 512],
                 hidden_channels=256,
                 embed_dims=512,
                 num_heads=8,
                 num_points=4,
                 feedforward_channels=1024,
                 dropout=0.1,
                 norm_cfg=dict(type='GN', num_groups=32),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims

        # 1. Input Projections
        self.input_projs = nn.ModuleList()
        for ch in in_channels:
            self.input_projs.append(
                nn.Sequential(
                    nn.Conv2d(ch, embed_dims, 1),
                    build_norm_layer(norm_cfg, embed_dims)[1]
                )
            )

        # 2. Level Embeddings
        self.level_embeds = nn.Parameter(torch.Tensor(len(in_channels), embed_dims))

        # 3. Deformable Attention
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_levels=len(in_channels),
            num_heads=num_heads,
            num_points=num_points,
            dropout=dropout,
            batch_first=True
        )

        # 4. FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.GELU(),
            nn.Linear(feedforward_channels, embed_dims)
        )

        # 5. Norms
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        # 6. Query Init Projection
        self.query_init_proj = nn.Conv2d(in_channels[0], embed_dims, 1)

    def init_weights(self):
        super().init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        xavier_init(self.level_embeds, distribution='uniform')

    def get_reference_points_and_query(self, visual_masks, p3_feat):
        """
        改进版：寻找 Mask 区域内特征响应最强的点作为参考点，并采样该点的特征作为 Query。
        解决 '重心落在背景' 和 '平均特征模糊' 的问题。

        Args:
            visual_masks: (B, N, H, W)
            p3_feat: (B, C, H, W) - 已经投影到 embed_dims 的特征
        Returns:
            ref_points: (B, N, 1, 2) normalized [0, 1]
            query_init: (B, N, C)
        """
        B, N, H, W = visual_masks.shape

        # 1. 计算特征的响应强度 (L2 Norm) -> (B, 1, H, W)
        # 我们假设模长越大的点，包含的信息量越大（前景概率越高）
        feature_norm = torch.norm(p3_feat, dim=1, keepdim=True)

        # 2. 结合 Mask: 只保留 Mask 内部的响应
        # visual_masks (B, N, H, W) * feature_norm (B, 1, H, W) -> (B, N, H, W)
        masked_response = visual_masks * feature_norm

        # 3. 展平空间维度: (B, N, H*W)
        flatten_response = masked_response.flatten(2)

        # 4. 找到最大响应值的索引 (Argmax)
        # max_idx: (B, N) - 每个 mask 内部最强点的索引 (0 ~ H*W-1)
        max_val, max_idx = flatten_response.max(dim=-1)

        # 5. 将索引转换为 (x, y) 坐标
        # y = idx // W, x = idx % W
        ref_y = max_idx // W
        ref_x = max_idx % W

        # 归一化到 [0, 1]
        # +0.5 是为了取像素中心，更精确
        ref_y_norm = (ref_y.float() + 0.5) / H
        ref_x_norm = (ref_x.float() + 0.5) / W

        # 堆叠参考点: (B, N, 1, 2) -> (x, y)
        ref_points = torch.stack([ref_x_norm, ref_y_norm], dim=-1).unsqueeze(2)

        # ---------------------------------------------------------
        # 6. 获取 Query Init: 使用参考点处的特征，而不是全局平均
        # ---------------------------------------------------------
        # grid_sample 需要坐标在 [-1, 1] 之间
        # ref_points 是 [0, 1]，转换公式: grid = points * 2 - 1
        sampling_grid = ref_points.squeeze(2).unsqueeze(1) * 2 - 1  # (B, 1, N, 2)

        # 采样特征: (B, C, 1, N)
        # align_corners=False 与上面的 +0.5 逻辑对应
        sampled_feat = F.grid_sample(p3_feat, sampling_grid, align_corners=False, mode='bilinear')

        # 调整形状: (B, C, 1, N) -> (B, C, N) -> (B, N, C)
        query_init = sampled_feat.squeeze(2).permute(0, 2, 1)

        return ref_points.clamp(0.01, 0.99), query_init

    def forward(self, x, vp):
        """
        x: [P3, P4, P5]
        vp: Visual Masks
        """
        # x[0] 是 P3, 形状 (B, C0, H, W)
        B, N, H, W = vp.shape

        # -----------------------------------------------------------
        # 1. 准备 Multi-scale Key/Value (保持不变)
        # -----------------------------------------------------------
        src_flattens = []
        spatial_shapes = []
        for idx, feat in enumerate(x):
            src = self.input_projs[idx](feat)
            bs, c, h_i, w_i = src.shape
            spatial_shapes.append((h_i, w_i))
            src = src.flatten(2).transpose(1, 2)
            src = src + self.level_embeds[idx].view(1, 1, -1)
            src_flattens.append(src)

        src_flattens = torch.cat(src_flattens, 1)
        spatial_shapes = torch.tensor(spatial_shapes, device=vp.device, dtype=torch.long)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # -----------------------------------------------------------
        # 2. 改进的初始化逻辑 (替换了原来的 Mask Pooling)
        # -----------------------------------------------------------
        p3_feat = x[0]
        p3_proj = self.query_init_proj(p3_feat)  # (B, C, H, W)

        # 同时获取 参考点 和 初始Query (使用 Argmax 逻辑)
        reference_points, query_init = self.get_reference_points_and_query(vp, p3_proj)

        # -----------------------------------------------------------
        # 3. Deformable Attention (保持不变)
        # -----------------------------------------------------------
        query_interact = self.cross_attn(
            query=query_init,
            value=src_flattens,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        query_out = self.norm1(query_init + query_interact)
        query_final = self.norm2(query_out + self.ffn(query_out))

        return F.normalize(query_final, dim=-1, p=2)


# class OrthogonalFusionModule(nn.Module):
#     """
#     正交特征融合模块 (OPR)
#     """
#
#     def __init__(self, embed_dims=512):
#         super().__init__()
#
#         # 1. Adapter: 将 Visual 空间映射到 Text 空间
#         self.visual_adapter = nn.Sequential(
#             nn.Linear(embed_dims, embed_dims),
#             nn.LayerNorm(embed_dims),
#             nn.SiLU(),
#             nn.Linear(embed_dims, embed_dims)
#         )
#
#         # 2. Gating Network: 用于决定注入多少 "个性细节" (垂直分量)
#         self.detail_gate = nn.Sequential(
#             nn.Linear(embed_dims * 2, embed_dims),
#             nn.Sigmoid()
#         )
#
#         # 可学习的缩放因子
#         self.para_scale = nn.Parameter(torch.ones(1) * 0.5)
#
#     def forward(self, txt_feats, vis_feats):
#         # Alignment
#         v_proj = self.visual_adapter(vis_feats)
#
#         # Orthogonal Decomposition
#         t_norm = txt_feats.norm(dim=-1, keepdim=True) + 1e-6
#         t_unit = txt_feats / t_norm
#
#         projection_scalar = (v_proj * t_unit).sum(dim=-1, keepdim=True)
#         v_para = projection_scalar * t_unit
#         v_orth = v_proj - v_para
#
#         # Gated Recomposition
#         gate = self.detail_gate(torch.cat([txt_feats, v_orth], dim=-1))
#         fused_feats = txt_feats + (self.para_scale * v_para) + (gate * v_orth)
#
#         return fused_feats

class OrthogonalFusionModule(nn.Module):
    """
    正交特征融合模块 (OPR) V2.0
    改进点：
    1. 双门控机制：同时动态控制平行分量(共性)和垂直分量(细节)的注入比例。
    2. 返回投影后的视觉特征，以便计算对齐 Loss。
    """

    def __init__(self, embed_dims=512):
        super().__init__()

        # 1. Adapter: 将 Visual 空间映射到 Text 空间
        # 这是为了弥合 CNN/ViT 特征与 CLIP Text 特征的异构性
        self.visual_adapter = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.SiLU(),
            nn.Linear(embed_dims, embed_dims)
        )

        # 2. Detail Gate: 控制垂直分量 (个性/细节/噪声)
        # 输入: cat([txt, v_orth]) -> 决定还需要多少细节
        self.detail_gate = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.Sigmoid()
        )

        # 3. [新增] Parallel Gate: 控制平行分量 (视觉共性验证)
        # 替代了原来的 self.para_scale
        # 输入: cat([txt, v_proj]) -> 根据文本和视觉的匹配程度，决定视觉共性的权重
        # 逻辑: 对于 Rare 类，如果视觉特征很强且匹配，这个 Gate 应该变大
        self.para_gate = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.Sigmoid()
        )

        # self.out_norm = nn.LayerNorm(embed_dims)

    def forward(self, txt_feats, vis_feats):
        """
        Args:
            txt_feats: (B, C)
            vis_feats: (B, C) 来自 SAVPE 的原始输出
        Returns:
            fused_feats: (B, C)
            v_proj: (B, C) 经过适配器的视觉特征 (用于计算 Align Loss)
        """
        # --- Step 1: Alignment & Mapping ---
        # 必须先映射，再做分解
        v_proj = self.visual_adapter(vis_feats)

        # --- Step 2: Orthogonal Decomposition ---
        # 归一化文本向量作为基准
        t_norm = txt_feats.norm(dim=-1, keepdim=True) + 1e-6
        t_unit = txt_feats / t_norm

        # 计算投影 (平行分量 - 共性)
        # (v_proj · t_unit) * t_unit
        projection_scalar = (v_proj * t_unit).sum(dim=-1, keepdim=True)
        v_para = projection_scalar * t_unit

        # 计算残差 (垂直分量 - 细节)
        v_orth = v_proj - v_para

        # --- Step 3: Dual Gated Recomposition (双动态门控) ---

        # Gate 1: 平行门控 (决定视觉共性有多重要)
        # 它可以学习到：当 Text 很强时，是否还需要叠加 Visual 的共性？
        alpha_para = self.para_gate(torch.cat([txt_feats, v_proj], dim=-1))

        # Gate 2: 细节门控 (决定注入多少细节)
        # 它可以学习到：当 Visual 包含 Text 没有的信息时，打开门
        beta_orth = self.detail_gate(torch.cat([txt_feats, v_orth], dim=-1))

        # 最终融合
        # Base(Text) + Dynamic(Visual_Para) + Dynamic(Visual_Orth)
        fused_feats = txt_feats + (alpha_para * v_para) + (beta_orth * v_orth)

        fused_feats = F.normalize(fused_feats, dim=-1)

        return fused_feats, v_proj
