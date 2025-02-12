import matplotlib.pyplot as _plt
import numpy as _np
import torch as _torch
import torch.nn as _nn
from PIL import Image as _Image

import src.vision_transformer.model.base as _base


def display_tensor_mask(mask: _torch.Tensor) -> _Image:
    """
    Display the mask tensor as an image.

    :param mask: The mask tensor.
    """
    mask = mask.detach().cpu().squeeze(0).numpy()
    mask = (mask * 255).astype(_np.uint8)
    mask = _Image.fromarray(mask, mode='L')
    return mask


def display_tensor_image(image: _torch.Tensor) -> _Image:
    """
    Display the image tensor as an image.

    :param image: The image tensor.
    """
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(_np.uint8)
    image = _Image.fromarray(image)
    return image


def display_attention_weights(
        model: _base.SemanticSegmentationVisionTransformerBase,
        img_original: _torch.Tensor,
        img_pre: _torch.Tensor,
        patch_size: int,
        scale_key: str,
        layer: int,
        use_max_pooling: bool = False,
) -> None:
    """
    Function to visualize the attention of the model.

    :param model: The model to visualize.
    :param img_original: The original image.
    :param img_pre: The preprocessed image.
    :param patch_size: The patch size.
    :param scale_key: The scale key.
    :param layer: The layer to visualize.
    :param use_max_pooling: Whether to use max pooling.
    """
    attention = _get_attention_weights(model, img_pre, patch_size, scale_key, layer, use_max_pooling)
    _plot_attention(img_original, attention)


########################################################################################################################
# Private Helpers
########################################################################################################################


def _get_attention_weights(
        model: _base.SemanticSegmentationVisionTransformerBase,
        img: _torch.Tensor,
        patch_size: int,
        scale_key: str,
        layer: int,
        use_max_pooling: bool = False,
) -> _np.ndarray:
    """
    Function to visualize the attention of the model.

    :param model: The model to visualize.
    :param img: The image tensor.
    :param patch_size: The patch size.
    :param scale_key: The scale key.
    :param layer: The layer to visualize.
    :param use_max_pooling: Whether to use max pooling.
    :return: Attention map.
    """
    # make the image divisible by the patch size
    w, h = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    _, attentions = model(img, return_attention_weights=True)
    attentions = attentions[scale_key][layer]

    if attentions.dim() == 4:
        H = attentions.shape[1]  # number of head
        # keep only the output patch attention
        if use_max_pooling:
            attentions = attentions[0].max(dim=1).values
        else:
            attentions = attentions[0, :, 0, :].reshape(H, -1)

    elif attentions.dim() == 5:
        W, H, _, E = attentions.shape
        if use_max_pooling:
            attentions = attentions.max(dim=2).values
        else:
            attentions = attentions[:, :, 0, :]
        attentions = attentions.permute(1, 0, 2).reshape(H, W * E)

    attentions = attentions.reshape(H, w_featmap, h_featmap)
    attentions = (
        _nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .detach()
        .cpu()
        .numpy()
    )

    return attentions


def _plot_attention(img: _torch.Tensor, attention: _np.ndarray) -> None:
    """
    Function to plot the attention map.

    :param img: The image tensor.
    :param attention: The attention map.
    """
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(_np.uint8)
    img = _Image.fromarray(img)

    n_heads = attention.shape[0]

    _plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, _np.mean(attention, 0)]):
        _plt.subplot(1, 2, i + 1)
        _plt.imshow(fig, cmap="inferno")
        _plt.title(text[i])
    _plt.show()

    _plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        _plt.subplot(n_heads // 4, 4, i + 1)
        _plt.imshow(attention[i], cmap="inferno")
        _plt.title(f"Head n: {i + 1}")
    _plt.tight_layout()
    _plt.show()
