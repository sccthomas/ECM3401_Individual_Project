import math as _math
import typing as _t

import matplotlib.pyplot as _plt
import numpy as _np
import torch as _torch
import torch.nn as _nn
from PIL import Image as _Image

import src.vision_transformer.model as _model


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
        model: _model.SemanticSegmentationVisionTransformer,
        img_original: _torch.Tensor,
        img_pre: _torch.Tensor,
        patch_sizes: _t.List[int],
        use_max_pooling: bool = False,
) -> None:
    """
    Function to visualize the attention of the model.

    :param model: The model to visualize.
    :param img_original: The original image.
    :param img_pre: The preprocessed image.
    :param patch_sizes: The patch sizes to visualize.
    :param use_max_pooling: Whether to use max pooling.
    """
    img_pre = img_pre.unsqueeze(0)
    _ = model(img_pre, keep_attention_scores=True)
    attentions = model.get_attention_scores()

    img_original = img_original.detach().cpu().permute(1, 2, 0).numpy()
    img_original = (img_original * 255).astype(_np.uint8)
    img_original = _Image.fromarray(img_original)

    kwargs = {
        "img": img_pre,
        "use_max_pooling": use_max_pooling,
    }
    for scale, patch_size in zip(attentions.keys(), patch_sizes):
        processed_attention = _process_attention_scores(
            attentions=attentions[scale],
            patch_size=patch_size,
            **kwargs,
        )
        for stage, attention_group in processed_attention.items():
            for i, attention in enumerate(attention_group, start=1):
                label_prefix = f"Scale:{scale} - Stage:{stage} - "
                # If attention is not a list, wrap it in a list for uniform iteration.
                att_list = attention if isinstance(attention, list) else [attention]
                label_suffix = "Patch Fusion" if isinstance(attention, list) else "Layer"
                for att in att_list:
                    _plot_attention(img_original, att, f"{label_prefix}{label_suffix}:{i}")


########################################################################################################################
# Private Helpers
########################################################################################################################


def _process_attention_scores(
        img: _torch.Tensor,
        patch_size: int,
        attentions: _t.Dict[str, _t.List[_t.Union[_torch.Tensor, _t.List[_torch.Tensor]]]],
        use_max_pooling: bool = False,
) -> _t.Dict[str, _t.List[_t.Union[_np.ndarray, _t.List[_np.ndarray]]]]:
    """
    Function to visualize the attention of the model.

    :param img: The image tensor.
    :param patch_size: The patch size.
    :param use_max_pooling: Whether to use max pooling.
    :return: Attention map.
    """
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    kwargs = {
        'w_featmap': w_featmap,
        'h_featmap': h_featmap,
        'use_max_pooling': use_max_pooling,
    }
    for key in attentions.keys():
        for i, attention in enumerate(attentions[key]):
            if isinstance(attention, list):
                for j, att in enumerate(attention):
                    attentions[key][i][j] = _upsample_attention(att, patch_size, **kwargs)
            else:
                attentions[key][i] = _upsample_attention(attention, patch_size, **kwargs)

    return attentions


def _upsample_attention(
        attention: _torch.Tensor, patch_size: int, w_featmap: int, h_featmap: int, use_max_pooling: bool
) -> _np.ndarray:
    """

    :param attention: The attention tensor.
    :param patch_size: The patch size.
    :param w_featmap: The width of the feature map.
    :param h_featmap: The height of the feature map.
    :param use_max_pooling: Whether to use max pooling.
    :return: The upsampled attention map.
    """
    if attention.dim() == 4:
        H = attention.shape[1]  # number of head
        # keep only the output patch attention
        if use_max_pooling:
            attention = attention[0].max(dim=1).values
        else:
            attention = attention[0, :, 0, :].reshape(H, -1)

    elif attention.dim() == 5:
        W, H, _, E = attention.shape
        if use_max_pooling:
            attention = attention.max(dim=2).values
        else:
            attention = attention[:, :, 0, :]
        attention = attention.permute(1, 0, 2).reshape(H, W * E)

    attention = attention.reshape(H, w_featmap, h_featmap)
    attention = (
        _nn.functional.interpolate(
            attention.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .detach()
        .cpu()
        .numpy()
    )

    return attention


def _plot_attention(img: _Image.Image, attention: _np.ndarray, title: str) -> None:
    """
    Function to plot the attention map.

    :param img: The image tensor.
    :param attention: The attention map.
    :param title: The title of the plot.
    """
    n_heads = attention.shape[0]

    # Dynamically determine grid dimensions for the attention heads.
    # Use at least 2 columns so that the top row can show both "Original Image" and "Head Mean"
    attn_cols = max(2, _math.ceil(_math.sqrt(n_heads)))
    attn_rows = _math.ceil(n_heads / attn_cols)

    total_rows = attn_rows + 1  # additional top row for the original image and head mean
    total_cols = attn_cols

    fig, axes = _plt.subplots(total_rows, total_cols, figsize=(total_cols * 3, total_rows * 3))

    # Ensure axes is always a 2D array (handles cases when total_rows or total_cols equals 1)
    if total_rows == 1:
        axes = _np.array([axes])
    if total_cols == 1:
        axes = _np.array([[ax] for ax in axes])

    # Set a title for the entire figure.
    fig.suptitle(title, fontsize=16)

    # --- Top row: Original Image and Head Mean ---
    # Position 0,0: Original Image.
    axes[0, 0].imshow(img, cmap="inferno")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Position 0,1: Head Mean (if available).
    if total_cols > 1:
        axes[0, 1].imshow(_np.mean(attention, axis=0), cmap="inferno")
        axes[0, 1].set_title("Head Mean")
        axes[0, 1].axis("off")

    # Hide any remaining subplots in the top row.
    for c in range(2, total_cols):
        axes[0, c].axis("off")

    # --- Remaining rows: Attention Heads ---
    for i in range(n_heads):
        # Determine row and column for this head (offset row index by 1)
        r, c = divmod(i, attn_cols)
        r += 1
        axes[r, c].imshow(attention[i], cmap="inferno")
        axes[r, c].set_title(f"Head {i + 1}")
        axes[r, c].axis("off")

    # Hide any unused subplots in the attention heads grid.
    for r in range(1, total_rows):
        for c in range(total_cols):
            if (r - 1) * attn_cols + c >= n_heads:
                axes[r, c].axis("off")

    _plt.tight_layout(rect=[0, 0, 1, 0.95])
    _plt.show()
