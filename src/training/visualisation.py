import math as _math
import os as _os
import re
import typing as _t

import matplotlib.gridspec as _gridspec
import matplotlib.pyplot as _plt
import matplotlib.pyplot as plt
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
        path: _t.Optional[str] = None,
        name: _t.Optional[str] = None,
) -> None:
    """
    Function to visualize the attention of the model.

    :param image: The original image.
    :param attention_scores: The attention scores.
    :param use_max_pooling: Whether to use max pooling.
    :param path: The path to save the image.
    :param name: The name of the image.
    """
    H, W = image.size()[-2:]

    # Process the attention scores
    kwargs = {"H": H, "W": W, "use_max_pooling": use_max_pooling}
    attentions = {
        (scale, patch_size): _process_attention_scores(attentions=attentions_scale, patch_size=patch_size, **kwargs)
        for (scale, patch_size), attentions_scale in attention_scores.items()
    }

    # Plot the attention scores
    rows = [
        (scale, patch_size, stage, layer_dict)
        for (scale, patch_size), stage_dict in attentions.items()
        for stage, layer_dict in stage_dict.items()
    ]
    max_layers = max(len(layer_list) for (_, _, _, layer_list) in rows)
    n_rows_total = len(rows)
    fig = plt.figure(figsize=(max_layers * 10, n_rows_total * 5), facecolor="white")
    fig.suptitle("Attention Maps", fontsize=32, fontweight="bold")
    outer_grid = _gridspec._gridspec(n_rows_total, max_layers, wspace=0.1, hspace=0.05, figure=fig)

    for grid_row, (scale, patch_size, stage, layer_dict) in enumerate(rows):
        for layer_idx, layer in layer_dict.items():
            inner_grid = _gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer_grid[grid_row, layer_idx],
                                                           wspace=0.275, hspace=0.005)
            # Plot the attention scores
            for head_idx in range(layer.shape[0]):
                ax = fig.add_subplot(inner_grid[head_idx // 4, head_idx % 4])
                img = ax.imshow(layer[head_idx], cmap='inferno', aspect='equal')
                ax.set_title(f'Head {head_idx + 1}', fontsize=15, fontweight='bold')
                ax.axis('off')

                # Add vertical color bar to each subplot and replace values with 'High' and 'Low'
                cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.04, pad=0.02)
                cbar.set_ticks([layer[head_idx].min(), layer[head_idx].max()])
                cbar.set_ticklabels(['Low', 'High'], fontweight="bold", fontsize=8)

    if path is not None:
        _os.makedirs(f"{path}/attention_scores", exist_ok=True)
        fig.savefig(f"{path}/attention_scores/{name}.png")
    else:
        plt.show()


def display_attention_weights_summary(
        image: _torch.Tensor,
        attention_scores: _t.Dict[_t.Tuple[str, int], _t.Dict[str, _t.List[_torch.Tensor]]],
        head_indices: _t.Dict[str, _t.List[int]],
        use_max_pooling: bool = False,
        path: _t.Optional[str] = None,
        name: _t.Optional[str] = None,
) -> None:
    """
    Function to visualize the attention of the model.

    :param image: The original image.
    :param attention_scores: The attention scores.
    :param head_indices: Dictionary with the indexes of attention heads to include in the plot.
    :param use_max_pooling: Whether to use max pooling.
    :param path: The path to save the image.
    :param name: The name of the image.
    """
    H, W = image.size()[-2:]

    # Process the attention scores
    kwargs = {"H": H, "W": W, "use_max_pooling": use_max_pooling}
    attentions = {
        (scale, patch_size): _process_attention_scores(attentions=attentions_scale, patch_size=patch_size, **kwargs)
        for (scale, patch_size), attentions_scale in attention_scores.items()
    }

    # Plot the attention scores
    rows = [
        (scale, patch_size, stage, layer_dict)
        for (scale, patch_size), stage_dict in attentions.items()
        for stage, layer_dict in stage_dict.items()
    ]

    fig = plt.figure(figsize=(8, len(rows) * 5), facecolor="white")
    fig.suptitle("Attention Maps", fontsize=32, fontweight="bold")
    outer_grid = _gridspec._gridspec(len(rows), 1, wspace=0.1, hspace=0.2, figure=fig)

    for grid_row, (scale, patch_size, stage, layer_dict) in enumerate(rows):
        inner_grid = _gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_grid[grid_row, 0],
                                                       wspace=0.1, hspace=0.2)
        for layer_idx, layer in layer_dict.items():
            head_idx = head_indices[scale][layer_idx] - 1
            # Plot the attention scores for the given head
            ax = fig.add_subplot(inner_grid[layer_idx // 3, layer_idx % 3])
            img = ax.imshow(layer[head_idx], cmap='inferno', aspect='equal')
            ax.set_title(f'Layer {layer_idx + 1} - Head {head_idx + 1}', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Add vertical color bar to each subplot and replace values with 'High' and 'Low'
            cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.04, pad=0.02)
            cbar.set_ticks([layer[head_idx].min(), layer[head_idx].max()])
            cbar.set_ticklabels(['Low', 'High'], fontweight="bold", fontsize=8)

    if path is not None:
        _os.makedirs(f"{path}/attention_scores_summary", exist_ok=True)
        fig.savefig(f"{path}/attention_scores_summary/{name}.png")
    else:
        plt.show()


def display_training_metrics(file_name: str) -> None:
    """
    Plot the training metrics.

    :param file_name: The file name.
    """
    parsed_file = _parse_log_file(file_name)
    epochs = parsed_file[0]
    training_losses = parsed_file[1]
    validation_losses = parsed_file[2]
    training_dice_scores = parsed_file[3]
    validation_dice_scores = parsed_file[4]
    training_miou = parsed_file[5]
    validation_miou = parsed_file[6]

    if len(epochs) == 0:
        print("No epochs found in the log file.")
        return
    if len(validation_losses) == 0:
        print("No validation losses found in the log file.")
        return

    # Find the epoch where validation loss is lowest
    best_epoch_idx = validation_losses.index(min(validation_losses))
    best_epoch = epochs[best_epoch_idx]

    plt.figure(figsize=(20, 8))

    # Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, training_losses, label='Training Loss', marker='o')
    plt.plot(epochs, validation_losses, label='Validation Loss', marker='s')
    plt.axvline(x=best_epoch, linestyle='--', color='red', alpha=0.6)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid()

    # Annotate values at best validation loss epoch
    if training_losses[best_epoch_idx] > validation_losses[best_epoch_idx]:
        plt.annotate(f"{training_losses[best_epoch_idx]:.4f}",
                     (best_epoch, training_losses[best_epoch_idx]),
                     textcoords="offset points", xytext=(-15, 30), ha='center', fontsize=10, fontweight='bold')

        plt.annotate(f"{validation_losses[best_epoch_idx]:.4f}",
                     (best_epoch, validation_losses[best_epoch_idx]),
                     textcoords="offset points", xytext=(-15, -15), ha='center', fontsize=10, fontweight='bold')
    else:
        plt.annotate(f"{training_losses[best_epoch_idx]:.4f}",
                     (best_epoch, training_losses[best_epoch_idx]),
                     textcoords="offset points", xytext=(-15, -15), ha='center', fontsize=10, fontweight='bold')

        plt.annotate(f"{validation_losses[best_epoch_idx]:.4f}",
                     (best_epoch, validation_losses[best_epoch_idx]),
                     textcoords="offset points", xytext=(-15, 30), ha='center', fontsize=10, fontweight='bold')

    # Dice Score Plot
    if len(training_dice_scores) > 0:
        plt.subplot(1, 3, 2)
        plt.plot(epochs, training_dice_scores, label='Training Dice', marker='o')
        plt.plot(epochs, validation_dice_scores, label='Validation Dice', marker='s')
        plt.axvline(x=best_epoch, linestyle='--', color='red', alpha=0.6)
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.title('Dice Score Over Epochs')
        plt.legend()
        plt.grid()

        # Annotate values at best validation loss epoch
        if training_dice_scores[best_epoch_idx] > validation_dice_scores[best_epoch_idx]:
            plt.annotate(f"{training_dice_scores[best_epoch_idx]:.4f}",
                         (best_epoch, training_dice_scores[best_epoch_idx]),
                         textcoords="offset points", xytext=(-15, 20), ha='center', fontsize=10, fontweight='bold')

            plt.annotate(f"{validation_dice_scores[best_epoch_idx]:.4f}",
                         (best_epoch, validation_dice_scores[best_epoch_idx]),
                         textcoords="offset points", xytext=(-15, -30), ha='center', fontsize=10, fontweight='bold')
        else:
            plt.annotate(f"{training_dice_scores[best_epoch_idx]:.4f}",
                         (best_epoch, training_dice_scores[best_epoch_idx]),
                         textcoords="offset points", xytext=(-15, -30), ha='center', fontsize=10, fontweight='bold')

            plt.annotate(f"{validation_dice_scores[best_epoch_idx]:.4f}",
                         (best_epoch, validation_dice_scores[best_epoch_idx]),
                         textcoords="offset points", xytext=(-15, 10), ha='center', fontsize=10, fontweight='bold')

    # Mean IoU Plot
    if len(training_miou) > 0:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, training_miou, label='Training mIoU', marker='o')
        plt.plot(epochs, validation_miou, label='Validation mIoU', marker='s')
        plt.axvline(x=best_epoch, linestyle='--', color='red', alpha=0.6)
        plt.xlabel('Epochs')
        plt.ylabel('Mean IoU')
        plt.title('Mean IoU Over Epochs')
        plt.legend()
        plt.grid()

        # Annotate values at best validation loss epoch
        if training_miou[best_epoch_idx] > validation_miou[best_epoch_idx]:
            plt.annotate(f"{training_miou[best_epoch_idx]:.4f}",
                         (best_epoch, training_miou[best_epoch_idx]),
                         textcoords="offset points", xytext=(-15, 20), ha='center', fontsize=10, fontweight='bold')

            plt.annotate(f"{validation_miou[best_epoch_idx]:.4f}",
                         (best_epoch, validation_miou[best_epoch_idx]),
                         textcoords="offset points", xytext=(-15, -30), ha='center', fontsize=10, fontweight='bold')
        else:
            plt.annotate(f"{training_miou[best_epoch_idx]:.4f}",
                         (best_epoch, training_miou[best_epoch_idx]),
                         textcoords="offset points", xytext=(-15, -30), ha='center', fontsize=10, fontweight='bold')

            plt.annotate(f"{validation_miou[best_epoch_idx]:.4f}",
                         (best_epoch, validation_miou[best_epoch_idx]),
                         textcoords="offset points", xytext=(-15, 10), ha='center', fontsize=10, fontweight='bold')

    plt.show()


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


def _parse_log_file(
        file_name: str
) -> _t.Tuple[
    _t.List[int], _t.List[float], _t.List[float], _t.List[float], _t.List[float], _t.List[float], _t.List[float]
]:
    """
    Parse the log file.

    :param file_name: The file name.
    :return: The parsed log file.
    """
    epochs = []
    training_losses = []
    validation_losses = []
    training_dice_scores = []
    validation_dice_scores = []
    training_miou = []
    validation_miou = []

    with open(file_name, 'r') as file:
        lines = file.readlines()

    current_phase = None  # Track whether we're in Training or Validation section

    for line in lines:
        if "Early stopping triggered" in line:
            break
        epoch_match = re.match(r"\s*Epoch (\d+)/", line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))
            continue

        if "Training Metrics" in line or "Training:" in line:
            current_phase = "training"
            continue
        elif "Validation Metrics" in line or "Validation:" in line:
            current_phase = "validation"
            continue

        loss_match = re.match(
            r"\s*(?:Average Binary Cross Entropy Loss|Average Loss|Training Loss|Validation Loss): ([0-9\.]+)",
            line
        )
        dice_match = re.match(r"\s*Average Dice Score: ([0-9\.]+)", line)
        miou_match = re.match(r"\s*Average Mean IoU: ([0-9\.]+)", line)

        if loss_match:
            if current_phase == "training":
                training_losses.append(float(loss_match.group(1)))
            elif current_phase == "validation":
                validation_losses.append(float(loss_match.group(1)))
        elif dice_match:
            if current_phase == "training":
                training_dice_scores.append(float(dice_match.group(1)))
            elif current_phase == "validation":
                validation_dice_scores.append(float(dice_match.group(1)))
        elif miou_match:
            if current_phase == "training":
                training_miou.append(float(miou_match.group(1)))
            elif current_phase == "validation":
                validation_miou.append(float(miou_match.group(1)))

    return (
        epochs,
        training_losses,
        validation_losses,
        training_dice_scores,
        validation_dice_scores,
        training_miou,
        validation_miou
    )
