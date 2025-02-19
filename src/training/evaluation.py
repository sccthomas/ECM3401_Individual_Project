import random as _random

import PIL.ImageFilter as _ImageFilter
import cv2 as _cv2
import matplotlib.pyplot as _plt
import numpy as _np
import torch as _torch
import torch.nn as _nn
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
) -> None:
    """
    Evaluate the model with texture modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Plot the images
    # - Show the unmodified image
    _visualisation.display_tensor_image(image)

    # Test the robustness of the model
    # - Normalise the image
    image = _transforms.Normalize(mean=_snow.MEAN, std=_snow.STD)(image)
    # - Evaluate the model
    outputs = model(image.unsqueeze(0).to(device))
    criterion = _nn.BCEWithLogitsLoss()
    loss = criterion(outputs, mask.unsqueeze(0)).item()
    print("BCE Loss:", loss)

    # Display the masks
    outputs = _torch.sigmoid(outputs).squeeze(0)
    print("Target Mask")
    _visualisation.display_tensor_mask(mask)
    print("Predicted Mask")
    _visualisation.display_tensor_mask(outputs > 0.5)
    _plt.show()


def evaluate_with_texture_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        texture_type: str = "Staining",
) -> None:
    """
    Evaluate the model with texture modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param texture_type: The type of texture modification. Options are:
        "Staining", "Background Artifacts", "Microscopic Artifacts", "Cellular Variability".
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

    _evaluate(model, image, image_modified, mask, device)


def evaluate_with_illumination_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        per_pixel: bool = False,
) -> None:
    """
    Evaluate the model with illumination modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param per_pixel: Whether to modify the image per pixel.
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

    _evaluate(model, image, image_modified, mask, device)


def evaluate_with_background_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        mtype: str = "Simple",
) -> None:
    """
    Evaluate the model with background modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param mtype: The type of background modification. Options are: "Simple", "Gaussian", "Contrast".
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

    _evaluate(model, image, image_modified, mask, device)


# --------------------------------------------
# Private Helpers
# --------------------------------------------

def _evaluate(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        image_modified: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
) -> None:
    """
    Evaluate the model with the modified image

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param image_modified: The modified image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    """
    # Plot the images
    # - Show the unmodified image
    print("Original Image")
    _visualisation.display_tensor_image(image)
    # - Show the modified image
    print("Modified Image")
    _visualisation.display_tensor_image(image_modified)

    # Test the robustness of the model
    # - Normalise the image
    image_modified = _transforms.Normalize(mean=_snow.MEAN, std=_snow.STD)(image_modified)
    # - Evaluate the model
    outputs = model(image_modified.unsqueeze(0).to(device))
    criterion = _nn.BCEWithLogitsLoss()
    loss = criterion(outputs, mask.unsqueeze(0)).item()
    print("BCE Loss:", loss)

    # Display the masks
    outputs = _torch.sigmoid(outputs).squeeze(0)
    print("Target Mask")
    _visualisation.display_tensor_mask(mask)
    print("Predicted Mask")
    _visualisation.display_tensor_mask(outputs > 0.5)


# --------------------------------------------
# Private Constants
# --------------------------------------------


_MEAN = [0.4808, 0.4178, 0.5046]
_STD = [0.2637, 0.2751, 0.2425]
