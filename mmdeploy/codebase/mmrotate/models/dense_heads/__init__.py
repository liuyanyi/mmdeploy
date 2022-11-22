# Copyright (c) OpenMMLab. All rights reserved.
# from .oriented_rpn_head import oriented_rpn_head__get_bboxes
# from .rotated_anchor_head import rotated_anchor_head__get_bbox
# from .rotated_rpn_head import rotated_rpn_head__get_bboxes
from .rotated_rtmdet_head import rtmdet_head__predict_by_feat

__all__ = [
    # 'oriented_rpn_head__get_bboxes', 'rotated_anchor_head__get_bbox',
    # 'rotated_rpn_head__get_bboxes'
    'rtmdet_head__predict_by_feat'
]
