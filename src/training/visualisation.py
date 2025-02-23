import math as _math
import typing as _t

import matplotlib.gridspec as _gridspec
import matplotlib.pyplot as _plt
import numpy as _np
import torch as _torch
import torch.nn as _nn


def display_tensor_mask(mask: _torch.Tensor, ax: _t.Optional[_plt.Axes] = None) -> None:
    """
    Display the mask tensor as an image.

    :param mask: The mask tensor.
    :param ax: Optional matplotlib Axes object to plot on.
    """
    mask = mask.detach().cpu().squeeze(0).numpy()
    mask = (mask * 255).astype(_np.uint8)
    is_ax = ax is None
    if is_ax:
        fig, ax = _plt.subplots()
    ax.imshow(mask, cmap='gray', aspect="equal")  # Use grayscale colormap
    ax.axis('off')  # Hide axis
    if is_ax:
        _plt.show()


def display_tensor_image(image: _torch.Tensor, ax: _t.Optional[_plt.Axes] = None) -> None:
    """
    Display the image tensor as an image.

    :param image: The image tensor.
    :param ax: Optional matplotlib Axes object to plot on.
    """
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(_np.uint8)
    is_ax = ax is None
    if is_ax:
        fig, ax = _plt.subplots()
    ax.imshow(image, aspect="equal")
    ax.axis('off')
    if is_ax:
        _plt.show()


def display_attention_weights(
        image: _torch.Tensor,
        attention_scores: _t.Dict[_t.Tuple[str, int], _t.Dict[str, _t.List[_torch.Tensor]]],
        use_max_pooling: bool = False,
) -> None:
    """
    Function to visualize the attention of the model.

    :param image: The original image.
    :param attention_scores: The attention scores.
    :param use_max_pooling: Whether to use max pooling.
    """
    H, W = image.size()[-2:]

    # Process the attention scores
    kwargs = {"H": H, "W": W, "use_max_pooling": use_max_pooling, }
    attentions = {
        (scale, patch_size): _process_attention_scores(attentions=attentions_scale, patch_size=patch_size, **kwargs)
        for (scale, patch_size), attentions_scale in attention_scores.items()
    }

    # Plot the attention scores
    rows = [
        (scale, patch_size, stage, layer_list)
        for (scale, patch_size), stage_dict in attentions.items()
        for stage, layer_list in stage_dict.items()
    ]
    max_layers = max(len(layer_list) for (_, _, _, layer_list) in rows)
    n_rows_total = len(rows)
    fig = _plt.figure(figsize=(max_layers * 15, n_rows_total * 10), facecolor="white")
    fig.suptitle("Attention Maps", fontsize=32, fontweight="bold")
    outer_grid = _gridspec.GridSpec(n_rows_total, max_layers, hspace=0.01, wspace=0.05, figure=fig)

    for grid_row, (scale, patch_size, stage, layer_list) in enumerate(rows):
        num_layers = len(layer_list)

        for l in range(max_layers):
            if l < num_layers:
                attn = layer_list[l]
                num_heads = attn.shape[0]
                nested = _gridspec.GridSpecFromSubplotSpec(
                    3, 4,
                    subplot_spec=outer_grid[grid_row, l],
                    height_ratios=[15, 7.5, 7.5], hspace=0.075, wspace=0.015
                )
                ax_avg = fig.add_subplot(nested[0, :])
                ax_avg.set_facecolor("white")
                avg_attn = attn.mean(axis=0)
                im_avg = ax_avg.imshow(avg_attn, aspect="equal", cmap="inferno")
                ax_avg.set_title(
                    f"{scale} | Patch Size {patch_size} | {stage} | Layer {l + 1}\nAvg",
                    fontsize=20, fontweight="bold"
                )
                ax_avg.axis("off")
                _plt.colorbar(im_avg, ax=ax_avg, fraction=0.02, pad=0.015)

                row = 1
                for h in range(num_heads):
                    ax_head = fig.add_subplot(nested[row, h % 4])
                    ax_head.set_facecolor("white")
                    im_head = ax_head.imshow(attn[h], aspect="equal", cmap="inferno")
                    ax_head.set_title(f"Head {h}", fontsize=16)
                    ax_head.axis("off")
                    if h % 4 == 3:
                        row += 1
            else:
                ax_dummy = fig.add_subplot(outer_grid[grid_row, l])
                ax_dummy.set_facecolor("white")
                ax_dummy.axis("off")

    _plt.show()


########################################################################################################################
# Private Helpers
########################################################################################################################


def _process_attention_scores(
        attentions: _t.Dict[str, _t.List[_torch.Tensor]],
        patch_size: int,
        H: int,
        W: int,
        use_max_pooling: bool = False,
) -> _t.Dict[str, _t.Dict[int, _np.ndarray]]:
    """
    Function to process the attention scores.

    :param attentions: The attention scores.
    :param patch_size: The patch size.
    :param H: The height of the image.
    :param W: The width of the image.
    :param use_max_pooling: Whether to use max pooling.
    :return: Attention map.
    """
    kwargs = {'w_featmap': W // patch_size, 'h_featmap': H // patch_size, 'use_max_pooling': use_max_pooling}
    attentions = {
        stage: {
            layer: _upsample_attention(attention_tensor, patch_size, **kwargs)
            for layer, attention_tensor in enumerate(attentions_stage)
            if isinstance(attention_tensor, _torch.Tensor)
        }
        for stage, attentions_stage in attentions.items()
    }

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
    # Normal attention scores of shape (B, Num_Heads, Num_Patches, Num_Patches)
    if attention.size(0) == 1:
        num_heads = attention.shape[1]
        if use_max_pooling:
            attention = attention[0].max(dim=1).values
        else:
            attention = attention[0, :, 0, :]
        attention = attention.reshape(num_heads, w_featmap, h_featmap)

    # Swin Transformer attention scores of shape (B * Num_Windows, Num_Heads, Num_Patches, Num_Patches)
    elif attention.size(0) > 1:
        num_windows, num_heads, N, _ = attention.shape
        window_size = int(_math.sqrt(N))

        if use_max_pooling:
            attention = attention.max(dim=2).values
        else:
            attention = attention[:, :, 0, :]

        attention = attention.reshape(num_windows, num_heads, window_size, window_size)
        grid_w = w_featmap // window_size
        grid_h = h_featmap // window_size
        assert grid_w * grid_h == num_windows, "Mismatch between window grid and number of windows"

        attention = attention.reshape(grid_h, grid_w, num_heads, window_size, window_size)
        attention = attention.permute(2, 0, 3, 1, 4)
        attention = attention.reshape(num_heads, grid_h * window_size, grid_w * window_size)

    attention = (
        _nn.functional.interpolate(
            attention.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .detach()
        .cpu()
        .numpy()
    )

    return attention
