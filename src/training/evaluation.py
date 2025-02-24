import random as _random

import cv2 as _cv2
import matplotlib.pyplot as _plt
import numpy as _np
import torch as _torch
import torch.nn as _nn
import torchmetrics.segmentation as _metrics
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
        normalise: bool = True,
        path: str = None,
        name: str = None,
) -> None:
    """
    Evaluate the model with texture modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    :param normalise: Whether to normalise the image.
    :param path: The path to save the figure.
    :param name: The name of the figure.
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
    image_ = _transforms.Normalize(mean=_snow.MEAN, std=_snow.STD)(image) if normalise else image
    outputs = model(image_.unsqueeze(0).to(device), keep_attention_scores=True)
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

    if path is not None:
        _plt.savefig(f"{path}/predictions/{name}.png")
    else:
        _plt.show()

    _visualisation.display_attention_weights(
        image=image,
        attention_scores=model.get_attention_scores(),
        use_max_pooling=use_max_pooling,
        path=path,
        name=name,
    )


def evaluate_with_texture_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        texture_type: str = "Staining",
        use_max_pooling: bool = True,
        normalise: bool = True,
        path: str = None,
        name: str = None,
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
    :param normalise: Whether to normalise the image.
    :param path: The path to save the figure.
    :param name: The name of the figure.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Modify and image
    H, W = image.shape[-2:]
    if texture_type == "Staining":
        factor = _np.random.uniform(0.8, 1.5)
        color_shift = _np.random.uniform(0.8, 1.2, size=(3, 1, 1))
        noise_texture = (image * factor * _torch.tensor(color_shift, device=device, dtype=image.dtype))
        noise_texture = noise_texture.clamp(0, 1)

    elif texture_type == "Background Artifacts":
        noise = _np.random.normal(loc=0, scale=0.1, size=(128, 128))
        noise = _cv2.resize(noise, (H, W), interpolation=_cv2.INTER_CUBIC)
        noise_texture = _torch.tensor(noise, device=device, dtype=image.dtype).squeeze(0).repeat(3, 1, 1)
        noise_texture = noise_texture + (_torch.rand_like(noise_texture) * 0.1)

    elif texture_type == "Microscopic Artifacts":
        artifact = _np.random.normal(loc=0, scale=0.05, size=(128, 128))
        artifact = _cv2.resize(artifact, (H, W), interpolation=_cv2.INTER_CUBIC)
        noise_texture = _torch.tensor(artifact, device=device, dtype=image.dtype).squeeze(0).repeat(3, 1, 1)
        gradient = _np.linspace(0, 1, H).reshape(-1, 1)
        gradient = _torch.tensor(gradient, device=device, dtype=image.dtype)
        noise_texture = noise_texture * gradient

    elif texture_type == "Cellular Variability":
        cell_size = _np.random.randint(5, 15)
        cells = _np.random.randint(0, 2, (H // cell_size, W // cell_size))
        cells = _cv2.resize(cells.astype(_np.float32), (H, W), interpolation=_cv2.INTER_NEAREST)
        cells = cells + (_np.random.randn(H, W) * 0.05)
        noise_texture = _torch.tensor(cells, device=device, dtype=image.dtype).squeeze(0).repeat(3, 1, 1)

    else:
        raise ValueError("Invalid type")
    new_background = (1 - mask) * noise_texture
    image_modified = mask * image + new_background

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title=f"Evaluation of the Vision Transformer With Texture Modification: {texture_type}",
        use_max_pooling=use_max_pooling,
        normalise=normalise,
        path=path,
        name=name,
    )


def evaluate_with_illumination_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        use_max_pooling: bool = True,
        normalise: bool = True,
        path: str = None,
        name: str = None,
) -> None:
    """
    Evaluate the model with illumination modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param use_max_pooling: Whether to use max pooling to downsample the attention
    :param normalise: Whether to normalise the image.
    :param path: The path to save the figure.
    :param name: The name of the figure.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Modify and image
    brightened_tensor = _transforms.ColorJitter(brightness=_random.uniform(0.5, 1.5))(image)
    new_background = (1 - mask) * brightened_tensor
    image_modified = mask * image + new_background

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title=f"Evaluation of the Vision Transformer With Illumination Modification",
        use_max_pooling=use_max_pooling,
        normalise=normalise,
        path=path,
        name=name,
    )


def evaluate_with_background_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        mtype: str = "Simple",
        use_max_pooling: bool = True,
        normalise: bool = True,
        path: str = None,
        name: str = None,
) -> None:
    """
    Evaluate the model with background modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param mtype: The type of background modification. Options are: "Simple", "Gaussian Blur", "Gaussian Noise", "Contrast", "Invert".
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    :param normalise: Whether to normalise the image.
    :param path: The path to save the figure.
    :param name: The name of the figure.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Modify and image
    if mtype == "Simple":
        modified_tensor = _torch.rand(3, 1, 1).to(device)
    elif mtype == "Gaussian Blur":
        modified_tensor = _transforms.GaussianBlur(kernel_size=5, sigma=1.5)(image)
    elif mtype == "Gaussian Noise":
        modified_tensor = _transforms.GaussianNoise(mean=0.5, sigma=0.25)(image)
    elif mtype == "Contrast":
        modified_tensor = _transforms.ColorJitter(contrast=_random.uniform(0.5, 1.5))(image)
    elif mtype == "Invert":
        modified_tensor = _transforms.RandomInvert(p=1)(image)
    elif mtype == "Sharpness":
        modified_tensor = _transforms.RandomAdjustSharpness(
            sharpness_factor=0, p=1
        )(image)
    else:
        raise ValueError("Invalid type")
    new_background = (1 - mask) * modified_tensor
    image_modified = mask * image + new_background

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title=f"Evaluation of the Vision Transformer With Background Modification: {mtype}",
        use_max_pooling=use_max_pooling,
        normalise=normalise,
        path=path,
        name=name,
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
        normalise: bool = True,
        path: str = None,
        name: str = None,
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
    :param normalise: Whether to normalise the image.
    :param path: The path to save the figure.
    :param name: The name of the figure.
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
    image_modified_ = (
        _transforms.Normalize(mean=_snow.MEAN, std=_snow.STD)(image_modified) if normalise else image_modified
    )
    # Evaluate the model.
    outputs = model(image_modified_.unsqueeze(0).to(device), keep_attention_scores=True)
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

    if path is not None:
        _plt.savefig(f"{path}/predictions/{name}.png")
    else:
        _plt.show()

    _visualisation.display_attention_weights(
        image=image,
        attention_scores=model.get_attention_scores(),
        use_max_pooling=use_max_pooling,
        path=path,
        name=name,
    )
