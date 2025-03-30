import os as _os
import random as _random
import typing as _t

import matplotlib.pyplot as _plt
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
        head_indices: _t.Optional[_t.Dict[str, _t.List[int]]] = None,
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
    :param head_indices: The indices of the heads to visualise.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Create a figure with 4 subplots in one row.
    fig, axes = _plt.subplots(1, 3, figsize=(12, 5))

    # Display the unmodified image.
    axes[0].set_title("Original Image", fontsize=15, fontweight='bold')
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
        f"BCE Loss: {bce_loss:.4f} | Dice Score {dice_loss:.4f} | IoU Score {IoU_loss:.4f}",
        fontsize=20, fontweight='bold'
    )

    # Display the masks
    outputs = _torch.sigmoid(outputs).squeeze(0)
    axes[1].set_title("Target Mask", fontsize=15, fontweight='bold')
    _visualisation.display_tensor_mask(mask.squeeze(0), ax=axes[1])
    axes[2].set_title("Predicted Mask", fontsize=15, fontweight='bold')
    _visualisation.display_tensor_mask(outputs > 0.5, ax=axes[2])

    _plt.tight_layout()

    if path is not None:
        path = f"{path}/no_modifications"
        _os.makedirs(f"{path}/predictions", exist_ok=True)
        _plt.savefig(f"{path}/predictions/{name}.png")
    else:
        _plt.show()

    if head_indices is not None:
        _visualisation.display_attention_weights_summary(
            image=image,
            attention_scores=model.get_attention_scores(),
            head_indices=head_indices,
            use_max_pooling=use_max_pooling,
            path=path,
            name=name,
        )
    else:
        _visualisation.display_attention_weights(
            image=image,
            attention_scores=model.get_attention_scores(),
            use_max_pooling=use_max_pooling,
            path=path,
            name=name,
        )


def evaluate_with_color_jitter(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        use_max_pooling: bool = True,
        normalise: bool = True,
        background_and_cells: bool = False,
        path: str = None,
        name: str = None,
        head_indices: _t.Optional[_t.Dict[str, _t.List[int]]] = None,
) -> None:
    """
    Evaluate the model with color jitter modifications.

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    :param normalise: Whether to normalise the image.
    :param background_and_cells: Whether to modify the background and cells together.
    :param path: The path to save the figure.
    :param name: The name of the figure.
    :param head_indices: The indices of the heads to visualise.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    transform_colour = _transforms.ColorJitter(
        contrast=_random.uniform(0.5, 1.5),
        saturation=_random.uniform(0.5, 1.5),
        hue=_random.uniform(0, 0.1)
    )
    jitter_tensor = transform_colour(image)
    new_background = (1 - mask) * jitter_tensor

    if background_and_cells:
        jitter_tensor = transform_colour(image)
        new_foreground = mask * jitter_tensor
        image_modified = new_foreground + new_background
        name = f"{name}_background_and_cells"
    else:
        image_modified = mask * image + new_background

    if path is not None:
        path = f"{path}/color_jitter"

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title="Evaluation with Color Jitter",
        use_max_pooling=use_max_pooling,
        normalise=normalise,
        path=path,
        name=name,
        head_indices=head_indices,
    )


def evaluate_with_noise_addition(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        use_max_pooling: bool = True,
        normalise: bool = True,
        background_and_cells: bool = False,
        path: str = None,
        name: str = None,
        head_indices: _t.Optional[_t.Dict[str, _t.List[int]]] = None,
) -> None:
    """
    Evaluate the model with noise added to the background.

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    :param normalise: Whether to normalise the image.
    :param background_and_cells: Whether to modify the background and cells together.
    :param path: The path to save the figure.
    :param name: The name of the figure.
    :param head_indices: The indices of the heads to visualise.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    noise_tensor = _transforms.GaussianNoise(mean=_snow.MEAN[0], sigma=_snow.STD[0])(image)
    new_background = (1 - mask) * noise_tensor

    if background_and_cells:
        noise_tensor = _transforms.GaussianNoise()(image)
        new_foreground = mask * noise_tensor
        image_modified = new_foreground + new_background
        name = f"{name}_background_and_cells"
    else:
        image_modified = mask * image + new_background

    if path is not None:
        path = f"{path}/noise_addition"

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title="Evaluation with Noise Addition",
        use_max_pooling=use_max_pooling,
        normalise=normalise,
        path=path,
        name=name,
        head_indices=head_indices,
    )


def evaluate_with_blur(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        use_max_pooling: bool = True,
        normalise: bool = True,
        background_and_cells: bool = False,
        path: str = None,
        name: str = None,
        head_indices: _t.Optional[_t.Dict[str, _t.List[int]]] = None,
) -> None:
    """
    Evaluate the model with blur modifications.

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    :param normalise: Whether to normalise the image.
    :param background_and_cells: Whether to modify the background and cells together.
    :param path: The path to save the figure.
    :param name: The name of the figure.
    :param head_indices: The indices of the heads to visualise.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    transform_blur = _transforms.GaussianBlur(kernel_size=7, sigma=(1, 2))
    blurred_tensor = transform_blur(image)
    new_background = (1 - mask) * blurred_tensor

    if background_and_cells:
        blurred_tensor = transform_blur(image)
        new_foreground = mask * blurred_tensor
        image_modified = new_foreground + new_background
        name = f"{name}_background_and_cells"
    else:
        image_modified = mask * image + new_background

    if path is not None:
        path = f"{path}/blur"

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title="Evaluation with Blur",
        use_max_pooling=use_max_pooling,
        normalise=normalise,
        path=path,
        name=name,
        head_indices=head_indices,
    )


def evaluate_with_synthetic_background(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        use_max_pooling: bool = True,
        normalise: bool = True,
        background_and_cells: bool = False,
        path: str = None,
        name: str = None,
        head_indices: _t.Optional[_t.Dict[str, _t.List[int]]] = None,
) -> None:
    """
    Evaluate the model with a synthetic plain background color.

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    :param normalise: Whether to normalise the image.
    :param background_and_cells: Whether to modify the background and cells together.
    :param path: The path to save the figure.
    :param name: The name of the figure.
    :param head_indices: The indices of the heads to visualise.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    background_color = _torch.tensor(
        [0.4, 0.6, 0.4],
        device=device,
        dtype=image.dtype,
    ).view(3, 1, 1)
    new_background = (1 - mask) * background_color

    if background_and_cells:
        blurred_tensor = _transforms.GaussianBlur(kernel_size=7, sigma=(1, 2))(image)
        new_foreground = mask * blurred_tensor
        image_modified = new_foreground + new_background
        name = f"{name}_background_and_cells"
    else:
        image_modified = mask * image + new_background

    if path is not None:
        path = f"{path}/synthetic_background"

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title="Evaluation with Synthetic Background",
        use_max_pooling=use_max_pooling,
        normalise=normalise,
        path=path,
        name=name,
        head_indices=head_indices,
    )


def evaluate_with_stain_variation(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        use_max_pooling: bool = True,
        normalise: bool = True,
        background_and_cells: bool = False,
        path: str = None,
        name: str = None,
        head_indices: _t.Optional[_t.Dict[str, _t.List[int]]] = None,
) -> None:
    """
    Evaluate the model with stain variation simulation.

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param use_max_pooling: Whether to use max pooling to downsample the attention scores.
    :param normalise: Whether to normalise the image.
    :param background_and_cells: Whether to modify the background and cells together.
    :param path: The path to save the figure.
    :param name: The name of the figure.
    :param head_indices: The indices of the heads to visualise.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    transform_stain_variation = _transforms.ColorJitter(
        brightness=_random.uniform(0.8, 1.2),
        contrast=_random.uniform(0.8, 1.2)
    )
    stain_variation = transform_stain_variation(image)
    new_background = (1 - mask) * stain_variation

    if background_and_cells:
        stain_variation = transform_stain_variation(image)
        new_foreground = mask * stain_variation
        image_modified = new_foreground + new_background
        name = f"{name}_background_and_cells"
    else:
        image_modified = mask * image + new_background

    if path is not None:
        path = f"{path}/stain_variation"

    _evaluate(
        model=model,
        image=image,
        image_modified=image_modified,
        mask=mask,
        device=device,
        title="Evaluation with Stain Variation",
        use_max_pooling=use_max_pooling,
        normalise=normalise,
        path=path,
        name=name,
        head_indices=head_indices,
    )


def evaluate_with_illumination_modifications(
        model: _model.SemanticSegmentationVisionTransformer,
        image: _torch.Tensor,
        mask: _torch.Tensor,
        device: _torch.device,
        use_max_pooling: bool = True,
        normalise: bool = True,
        background_and_cells: bool = False,
        path: str = None,
        name: str = None,
        head_indices: _t.Optional[_t.Dict[str, _t.List[int]]] = None,
) -> None:
    """
    Evaluate the model with illumination modifications

    :param model: The model to evaluate.
    :param image: The input image, shape (3, H, W).
    :param mask: The target mask, shape (1, H, W).
    :param device: The device to use.
    :param use_max_pooling: Whether to use max pooling to downsample the attention
    :param normalise: Whether to normalise the image.
    :param background_and_cells: Whether to modify the background and cells together.
    :param path: The path to save the figure.
    :param name: The name of the figure.
    :param head_indices: The indices of the heads to visualise.
    """
    model = model.to(device).eval()
    image = image.to(device)
    mask = mask.to(device)

    # Modify and image
    transform_illumination = _transforms.ColorJitter(brightness=_random.uniform(0.5, 1.5))
    brightened_tensor = transform_illumination(image)
    new_background = (1 - mask) * brightened_tensor

    if background_and_cells:
        brightened_tensor = transform_illumination(image)
        new_foreground = mask * brightened_tensor
        image_modified = new_foreground + new_background
        name = f"{name}_background_and_cells"
    else:
        image_modified = mask * image + new_background

    if path is not None:
        path = f"{path}/illumination_modifications"

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
        head_indices=head_indices,
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
        head_indices: _t.Optional[_t.Dict[str, _t.List[int]]] = None,
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
    :param head_indices: The indices of the heads to visual
    """
    # Create a figure with 4 subplots in one row.
    fig, axes = _plt.subplots(1, 8, figsize=(20, 5))

    # Display the unmodified image.
    axes[0].set_title("Original Image", fontsize=10)
    _visualisation.display_tensor_image(image, ax=axes[0])
    axes[1].set_title("Original Image - Background Context", fontsize=10)
    _visualisation.display_tensor_image(image * (1 - mask), ax=axes[1])
    axes[2].set_title("Original Image - Cells", fontsize=10)
    _visualisation.display_tensor_image(image * mask, ax=axes[2])

    # Display the modified image.
    axes[3].set_title("Modified Image", fontsize=10)
    _visualisation.display_tensor_image(image_modified, ax=axes[3])
    axes[4].set_title("Modified Image - Background Context", fontsize=10)
    _visualisation.display_tensor_image(image_modified * (1 - mask), ax=axes[4])
    axes[5].set_title("Modified Image - Cells", fontsize=10)
    _visualisation.display_tensor_image(image_modified * mask, ax=axes[5])

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
    fig.suptitle(f"{title} | BCE Loss: {bce_loss:.4f} | Dice Score {dice_loss:.4f} | IoU Score {IoU_loss:.4f}",
                 fontsize=16)

    # Display the masks
    axes[6].set_title("Target Mask", fontsize=10)
    _visualisation.display_tensor_mask(mask.squeeze(0), ax=axes[6])
    axes[7].set_title("Predicted Mask", fontsize=10)
    _visualisation.display_tensor_mask(outputs.squeeze(0), ax=axes[7])

    _plt.tight_layout()

    if path is not None:
        path_ = f"{path}/predictions"
        _os.makedirs(path_, exist_ok=True)
        _plt.savefig(f"{path_}/{name}.png")
    else:
        _plt.show()

    if head_indices is not None:
        _visualisation.display_attention_weights_summary(
            image=image,
            attention_scores=model.get_attention_scores(),
            head_indices=head_indices,
            use_max_pooling=use_max_pooling,
            path=path,
            name=name,
        )
    else:
        _visualisation.display_attention_weights(
            image=image,
            attention_scores=model.get_attention_scores(),
            use_max_pooling=use_max_pooling,
            path=path,
            name=name,
        )
