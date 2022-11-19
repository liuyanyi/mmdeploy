# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import Config, Registry
from mmengine.model import BaseDataPreprocessor
from mmengine.structures import BaseDataElement, InstanceData
from torch import Tensor, nn

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config, get_partition_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = Registry('backend_rotated_detectors')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of rotated detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        class_names (Sequence[str]): A list of string specifying class names.
        device (str): A string represents device type.
        deploy_cfg (str | mmengine.Config): Deployment config file or loaded
            Config object.
        model_cfg (str | mmengine.Config): Model config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, Config],
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 **kwargs):
        super().__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self.device = device
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        """Initialize backend wrapper.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
        """
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)

    @staticmethod
    def __clear_outputs(
        test_outputs: List[Union[Tensor, np.ndarray]]
    ) -> List[Union[List[Tensor], List[np.ndarray]]]:
        """Removes additional outputs and detections with zero and negative
        score.

        Args:
            test_outputs (List[Union[Tensor, np.ndarray]]):
                outputs of forward_test.

        Returns:
            List[Union[List[Tensor], List[np.ndarray]]]:
                outputs with without zero score object.
        """
        batch_size = len(test_outputs[0])

        num_outputs = len(test_outputs)
        outputs = [[None for _ in range(batch_size)]
                   for _ in range(num_outputs)]

        for i in range(batch_size):
            inds = test_outputs[0][i, :, -1] > 0.0
            for output_id in range(num_outputs):
                outputs[output_id][i] = test_outputs[output_id][i, inds, ...]
        return outputs

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict',
                **kwargs) -> Any:
        """The model forward.

        Args:
            inputs (torch.Tensor): The input tensors
            data_samples (List[BaseDataElement], optional): The data samples.
                Defaults to None.
            mode (str, optional): forward mode, only support `predict`.

        Returns:
            Any: Model output.
        """
        assert mode == 'predict', 'Deploy model only allow mode=="predict".'
        inputs = inputs.contiguous()
        outputs = self.predict(inputs)
        outputs = End2EndModel.__clear_outputs(outputs)
        batch_dets, batch_labels = outputs[:2]
        batch_size = inputs.shape[0]
        img_metas = [data_sample.metainfo for data_sample in data_samples]
        results = []
        rescale = kwargs.get('rescale', True)
        for i in range(batch_size):
            dets, labels = batch_dets[i], batch_labels[i]
            result = InstanceData()

            bboxes = dets[:, :5]
            scores = dets[:, 5]
            # perform rescale
            if rescale and 'scale_factor' in img_metas[i]:
                scale_factor = img_metas[i]['scale_factor']
                if isinstance(scale_factor, (list, tuple, np.ndarray)):
                    if len(scale_factor) == 2:
                        scale_factor = np.array(scale_factor)
                        scale_factor = np.concatenate(
                            [scale_factor, scale_factor])
                    scale_factor = np.array(scale_factor)[None, :]  # [1,4]
                scale_factor = torch.from_numpy(scale_factor).to(dets)
                dets[:, :4] /= scale_factor
            pad_key = None
            if 'pad_param' in img_metas[i]:
                pad_key = 'pad_param'
            elif 'border' in img_metas[i]:
                pad_key = 'border'
            if pad_key is not None:
                scale_factor = img_metas[i].get('scale_factor',
                                                np.array([1., 1.]))
                x_off = img_metas[i][pad_key][2] / scale_factor[1]
                y_off = img_metas[i][pad_key][0] / scale_factor[0]
                bboxes[:, 0] -= x_off
                bboxes[:, 1] -= y_off
                bboxes *= (bboxes > 0)

            # dets = dets.cpu().numpy()
            # labels = labels.cpu().numpy()
            # dets_results = [
            #     dets[labels == i, :] for i in range(len(self.CLASSES))
            # ]

            result.scores = scores
            result.bboxes = bboxes
            result.labels = labels
            data_samples[i].pred_instances = result

            results.append(data_samples[i])

        return results

    def predict(self, imgs: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """The interface for predict.

        Args:
            imgs (Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        return outputs


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmrotate format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img: List[torch.Tensor], *args, **kwargs) -> list:
        """Run forward inference.

        Args:
            img (List[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        results = []
        dets, labels = self.wrapper.invoke(
            img[0].contiguous().detach().cpu().numpy())
        dets_results = [dets[labels == i, :] for i in range(len(self.CLASSES))]
        results.append(dets_results)

        return results


def build_rotated_detection_model(
        model_files: Sequence[str],
        model_cfg: Union[str, Config],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build object detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model
        data_preprocessor (BaseDataPreprocessor | Config): The data
            preprocessor of the model.

    Returns:
        End2EndModel: Detector for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)

    partition_config = get_partition_config(deploy_cfg)
    if partition_config is not None:
        partition_type = partition_config.get('type', None)
    else:
        codebase_config = get_codebase_config(deploy_cfg)
        # Default Config is 'end2end'
        partition_type = codebase_config.get('model_type', 'end2end')

    backend_detector = __BACKEND_MODEL.build(
        dict(
            type=partition_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_detector
