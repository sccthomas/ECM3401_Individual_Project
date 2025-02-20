import random as _random

import PIL.ImageFilter as _ImageFilter
import cv2 as _cv2
import matplotlib.pyplot as _plt
import numpy as _np
import torch as _torch
import torch.nn as _nn
import torchmetrics.segmentation as _metrics
import torchvision.transforms.functional as _F
import torchvision.transforms.v2 as _transforms

import src.dataset.snow as _snow
import src.training.visualisation as _visualisation
import src.vision_transformer.model as _model


def evaluate_with_no_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        use_max_pooling: bool = True,
) -> None:
    """
    Evaluate the model with texture modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Create a figure with 4 subplots in one row.
    fig, axes = _plt.subplots(1, 3, figsize=(20, 5))

    # Display the unmodified image.
    axes[0].set_title("Original Image", fontsize=10)
    _visualisation.display_tensor_image(image, ax=axes[0])

    # Evaluate the model.
    image_norm = _transforms.Normalize(mean=_snow.MEAN, std=_snow.STD)(image)
    outputs = model(image_norm.unsqueeze(0).to(device), keep_attention_scores=True)
    # - Calculate the losses
    mask = mask.unsqueeze(0)
    #   - BCE Loss
    bce_loss = _nn.BCEWithLogitsLoss()(outputs, mask).item()
    #   - Dice & IoU Loss
    outputs = _torch.sigmoid(outputs) > 0.5
    dice_loss = _metrics.DiceScore(
        average='micro',
        num_classes=1,
    ).to(device)(outputs, mask).item()
    IoU_loss = _metrics.MeanIoU(
        num_classes=1,
    ).to(device)(outputs, mask.int()).item()
    # - Set the title
    fig.suptitle(
        f"Evaluation of the Vision Transformer With No Modifications | BCE Loss: {bce_loss:.4f} | Dice Loss {dice_loss:.4f} | IoU Loss {IoU_loss:.4f}",
        fontsize=16
    )

    # Display the masks
    outputs = _torch.sigmoid(outputs).squeeze(0)
    axes[1].set_title("Target Mask", fontsize=10)
    _visualisation.display_tensor_mask(mask.squeeze(0), ax=axes[1])
    axes[2].set_title("Predicted Mask", fontsize=10)
    _visualisation.display_tensor_mask(outputs > 0.5, ax=axes[2])

    _plt.tight_layout()
    _plt.show()

    _visualisation.display_attention_weights(
        image=image, attention_scores=model.get_attention_scores(), use_max_pooling=use_max_pooling
    )


def evaluate_with_texture_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        texture_type: str = "Staining",
        use_max_pooling: bool = True,
) -> None:
    """
    Evaluate the model with texture modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param texture_type: The type of texture modification. Options are:
        "Staining", "Background Artifacts", "Microscopic Artifacts", "Cellular Variability".
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Modify and image
    H, W = image.shape[-2:]
    if texture_type == "Staining":
        factor = _np.random.uniform(0.8, 1.5)
        noise_texture = (image * factor).clamp(0, 1)
    elif texture_type == "Background Artifacts":
        noise = _np.random.randn(128, 128) * 0.1  # Low-intensity random noise
        noise = _cv2.resize(noise, (H, W), interpolation=_cv2.INTER_CUBIC)
        noise_texture = _torch.tensor(noise, device=device, dtype=image.dtype).squeeze(0).repeat(3, 1, 1)
    elif texture_type == "Microscopic Artifacts":
        artifact = _np.random.randn(128, 128) * 0.05  # Smaller noise
        artifact = _cv2.resize(artifact, (H, W), interpolation=_cv2.INTER_CUBIC)
        noise_texture = _torch.tensor(artifact, device=device, dtype=image.dtype).squeeze(0).repeat(3, 1, 1)
    elif texture_type == "Cellular Variability":
        cell_size = 10  # Size of cellular structures
        cells = _np.random.randint(0, 2, (H // cell_size, W // cell_size))  # Random binary pattern
        cells = _cv2.resize(cells.astype(_np.float32), (H, W), interpolation=_cv2.INTER_NEAREST)
        noise_texture = _torch.tensor(cells, device=device, dtype=image.dtype).squeeze(0).repeat(3, 1, 1)
    else:
        raise ValueError("Invalid type")
    image_modified = mask * image + (1 - mask) * noise_texture

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title=f"Evaluation of the Vision Transformer With Texture Modification: {texture_type}",
        use_max_pooling=use_max_pooling
    )


def evaluate_with_illumination_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        per_pixel: bool = False,
        use_max_pooling: bool = True,
) -> None:
    """
    Evaluate the model with illumination modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param per_pixel: Whether to modify the image per pixel.
    :param use_max_pooling: Whether to use max pooling to downsample the attention
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Modify and image
    if per_pixel:
        gradient_map = _torch.linspace(0.8, 1.5, image.shape[-1]).to(device)
        gradient_map = gradient_map.view(1, 1, -1).repeat(1, image.shape[1], 1)
        gradient_map = gradient_map * (1 - mask)
        new_background = image * gradient_map

    else:
        brightness_factor = _random.uniform(0.8, 1.5)
        new_background = (1 - mask) * image * brightness_factor
    image_modified = mask * image + new_background

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title=f"Evaluation of the Vision Transformer With Illumination Modification: per_pixel = {per_pixel}",
        use_max_pooling=use_max_pooling
    )


def evaluate_with_background_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        mtype: str = "Simple",
        use_max_pooling: bool = True,
) -> None:
    """
    Evaluate the model with background modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param mtype: The type of background modification. Options are: "Simple", "Gaussian", "Contrast".
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Modify and image
    if mtype == "Simple":
        random_color = _torch.rand(3, 1, 1).to(device)
        new_background = (1 - mask) * random_color
    elif mtype == "Gaussian":
        img_pil = _F.to_pil_image(image)
        blurred_pil = img_pil.filter(_ImageFilter.GaussianBlur(radius=5))
        blurred_tensor = _F.to_tensor(blurred_pil).to(device)
        new_background = (1 - mask) * blurred_tensor
    elif mtype == "Contrast":
        contrast_factor = _random.uniform(0.5, 2.5)
        mean_intensity = image.mean(dim=(1, 2), keepdim=True)  # Compute per-channel mean
        new_background = (1 - mask) * (mean_intensity + contrast_factor * (image - mean_intensity))
    else:
        raise ValueError("Invalid type")
    image_modified = mask * image + new_background

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title=f"Evaluation of the Vision Transformer With Background Modification: {mtype}",
        use_max_pooling=use_max_pooling
    )


# --------------------------------------------
# Private Helpers
# --------------------------------------------

def _evaluate(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        image_modified: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        title: str,
        use_max_pooling: bool = True,
) -> None:
    """
    Evaluate the model with the modified image

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param image_modified: The modified image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param title: The title of the figure.
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    """
    # Create a figure with 4 subplots in one row.
    fig, axes = _plt.subplots(1, 4, figsize=(20, 5))

    # Display the unmodified image.
    axes[0].set_title("Original Image", fontsize=10)
    _visualisation.display_tensor_image(image, ax=axes[0])

    # Display the modified image.
    axes[1].set_title("Modified Image", fontsize=10)
    _visualisation.display_tensor_image(image_modified, ax=axes[1])

    # Test the robustness of the model.
    image_modified_norm = _transforms.Normalize(mean=_snow.MEAN, std=_snow.STD)(image_modified)
    # Evaluate the model.
    outputs = model(image_modified_norm.unsqueeze(0).to(device), keep_attention_scores=True)
    # - Calculate the losses
    mask = mask.unsqueeze(0)
    #   - BCE Loss
    bce_loss = _nn.BCEWithLogitsLoss()(outputs, mask).item()
    #   - Dice & IoU Loss
    outputs = _torch.sigmoid(outputs) > 0.5
    dice_loss = _metrics.DiceScore(
        average='micro',
        num_classes=1,
    ).to(device)(outputs, mask).item()
    IoU_loss = _metrics.MeanIoU(
        num_classes=1,
    ).to(device)(outputs, mask.int()).item()
    # - Set the title
    fig.suptitle(f"{title} | BCE Loss: {bce_loss:.4f} | Dice Loss {dice_loss:.4f} | IoU Loss {IoU_loss:.4f}",
                 fontsize=16)

    # Display the masks
    axes[2].set_title("Target Mask", fontsize=10)
    _visualisation.display_tensor_mask(mask.squeeze(0), ax=axes[2])
    axes[3].set_title("Predicted Mask", fontsize=10)
    _visualisation.display_tensor_mask(outputs.squeeze(0), ax=axes[3])

    _plt.tight_layout()
    _plt.show()

    _visualisation.display_attention_weights(
        image=image, attention_scores=model.get_attention_scores(), use_max_pooling=use_max_pooling
    )
