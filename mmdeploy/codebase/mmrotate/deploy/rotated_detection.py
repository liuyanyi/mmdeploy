# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Callable, Optional, Sequence

import torch
from mmengine import Config
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.codebase.mmdet import ObjectDetection
from mmdeploy.utils import Codebase, Task

MMROTATE_TASK = Registry('mmrotate_tasks')


@CODEBASE.register_module(Codebase.MMROTATE.value)
class MMROTATE(MMCodebase):
    """MMOCR codebase class."""

    task_registry = MMROTATE_TASK

    @classmethod
    def register_deploy_modules(cls):
        import mmdeploy.codebase.mmrotate.models  # noqa: F401

    @classmethod
    def register_all_modules(cls):
        from mmdet.utils.setup_env import \
            register_all_modules as register_all_modules_mmdet
        from mmrotate.utils.setup_env import \
            register_all_modules as register_all_modules_mmrotate

        from mmdeploy.codebase.mmdet.deploy.object_detection import MMDetection
        cls.register_deploy_modules()
        MMDetection.register_deploy_modules()
        register_all_modules_mmrotate(True)
        register_all_modules_mmdet(False)


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmrotate import datasets  # noqa
    from mmrotate.registry import DATASETS

    module_dict = DATASETS.module_dict

    for dataloader_name in [
        'test_dataloader', 'val_dataloader', 'train_dataloader'
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        dataset_cfg = dataloader_cfg.dataset
        dataset_cls = module_dict.get(dataset_cfg.type, None)
        if dataset_cls is None:
            continue
        if hasattr(dataset_cls, '_load_metainfo') and isinstance(
                dataset_cls._load_metainfo, Callable):
            meta = dataset_cls._load_metainfo(
                dataset_cfg.get('metainfo', None))
            if meta is not None:
                return meta
        if hasattr(dataset_cls, 'METAINFO'):
            return dataset_cls.METAINFO

    return None


@MMROTATE_TASK.register_module(Task.ROTATED_DETECTION.value)
class RotatedDetection(ObjectDetection):

    def build_backend_model(self,
                            model_files: Optional[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .rotated_detection_model import build_rotated_detection_model

        data_preprocessor = deepcopy(
            self.model_cfg.model.get('data_preprocessor', {}))
        data_preprocessor.setdefault('type', 'mmdet.DetDataPreprocessor')

        model = build_rotated_detection_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor)
        model = model.to(self.device)
        return model.eval()


    def get_visualizer(self, name: str, save_dir: str):
        from mmrotate.visualization import RotLocalVisualizer  # noqa: F401,F403
        metainfo = _get_dataset_metainfo(self.model_cfg)
        visualizer = super().get_visualizer(name, save_dir)
        if metainfo is not None:
            visualizer.dataset_meta = metainfo
        return visualizer
