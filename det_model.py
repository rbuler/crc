
from typing import Any, Sequence
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.utils.anchor_utils import AnchorGenerator
from monai.networks.nets import resnet
import logging
logger = logging.getLogger(__name__)


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
    return total_params

def retinanet_resnet_fpn_detector(
    num_classes: int,
    anchor_generator: AnchorGenerator,
    returned_layers: Sequence[int] = (1, 2, 3),
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> RetinaNetDetector:
    """
    Returns a RetinaNet detector using a ResNet-50 as backbone, which can be pretrained
    from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`
    _.

    Args:
        num_classes: number of output classes of the model (excluding the background).
        anchor_generator: AnchorGenerator,
        returned_layers: returned layers to extract feature maps. Each returned layer should be in the range [1,4].
            len(returned_layers)+1 will be the number of extracted feature maps.
            There is an extra maxpooling layer LastLevelMaxPool() appended.
        pretrained: If True, returns a backbone pre-trained on 23 medical datasets
        progress: If True, displays a progress bar of the download to stderr

    Return:
        A RetinaNetDetector object with resnet50 as backbone

    Example:

        .. code-block:: python

            # define a naive network
            resnet_param = {
                "pretrained": False,
                "spatial_dims": 3,
                "n_input_channels": 2,
                "num_classes": 3,
                "conv1_t_size": 7,
                "conv1_t_stride": (2, 2, 2)
            }
            returned_layers = [1]
            anchor_generator = monai.apps.detection.utils.anchor_utils.AnchorGeneratorWithAnchorShape(
                feature_map_scales=(1, 2), base_anchor_shapes=((8,) * resnet_param["spatial_dims"])
            )
            detector = retinanet_resnet50_fpn_detector(
                **resnet_param, anchor_generator=anchor_generator, returned_layers=returned_layers
            )
    Notes:

        Input argument ``network`` can be a monai.apps.detection.networks.retinanet_network.RetinaNet(*) object,
        but any network that meets the following rules is a valid input ``network``.

        1. It should have attributes including spatial_dims, num_classes, cls_key, box_reg_key, num_anchors, size_divisible.

            - spatial_dims (int) is the spatial dimension of the network, we support both 2D and 3D.
            - num_classes (int) is the number of classes, excluding the background.
            - size_divisible (int or Sequene[int]) is the expection on the input image shape.
              The network needs the input spatial_size to be divisible by size_divisible, length should be 2 or 3.
            - cls_key (str) is the key to represent classification in the output dict.
            - box_reg_key (str) is the key to represent box regression in the output dict.
            - num_anchors (int) is the number of anchor shapes at each location. it should equal to
              ``self.anchor_generator.num_anchors_per_location()[0]``.

        2. Its input should be an image Tensor sized (B, C, H, W) or (B, C, H, W, D).

        3. About its output ``head_outputs``:

            - It should be a dictionary with at least two keys:
              ``network.cls_key`` and ``network.box_reg_key``.
            - ``head_outputs[network.cls_key]`` should be List[Tensor] or Tensor. Each Tensor represents
              classification logits map at one resolution level,
              sized (B, num_classes*num_anchors, H_i, W_i) or (B, num_classes*num_anchors, H_i, W_i, D_i).
            - ``head_outputs[network.box_reg_key]`` should be List[Tensor] or Tensor. Each Tensor represents
              box regression map at one resolution level,
              sized (B, 2*spatial_dims*num_anchors, H_i, W_i)or (B, 2*spatial_dims*num_anchors, H_i, W_i, D_i).
            - ``len(head_outputs[network.cls_key]) == len(head_outputs[network.box_reg_key])``.
    """

    backbone = resnet.resnet18(pretrained, progress, **kwargs)
    # logger.info(f"backbone: {backbone}")
    spatial_dims = len(backbone.conv1.stride)
    # number of output feature maps is len(returned_layers)+1
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=spatial_dims,
        pretrained_backbone=pretrained,
        trainable_backbone_layers=None,
        returned_layers=returned_layers,
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    logger.info(f"num_anchors: {num_anchors}")
    size_divisible = [s * 2 * 2 ** max(returned_layers) for s in feature_extractor.body.conv1.stride]
    logger.info(f"size_divisible: {size_divisible}")
    network = RetinaNet(
        spatial_dims=spatial_dims,
        num_classes=num_classes,
        num_anchors=num_anchors,
        feature_extractor=feature_extractor,
        size_divisible=size_divisible,
    )
    logger.info(f"network params: {count_parameters(network)/1e6:.2f}M")
    return RetinaNetDetector(network, anchor_generator, debug=False)
