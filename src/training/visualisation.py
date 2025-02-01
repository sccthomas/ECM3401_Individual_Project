import cv2 as _cv2
import matplotlib.pyplot as _plt
import numpy as _np
import torch as _torch
from PIL import Image as _Image


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


def display_image_with_attention_heatmap(
        image_tensor: _torch.Tensor,
        attn_weights: _torch.Tensor,
        alpha: float = 0.5,
        cmap: str = 'jet',
) -> None:
    """
    Display an image with an attention heatmap overlay.

    :param image_tensor: The input image tensor.
    :param attn_weights: The attention weights tensor.
    :param alpha: The blending factor between the original image and the heatmap.
    :param cmap: The colormap to use for the heatmap.
    """
    # Convert the image tensor to a NumPy array.
    # Assuming image_tensor is in shape [C, W, H] (channel, width, height), we convert it to [W, H, C].
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)

    # Normalize image to [0, 1] if the pixel values are not already in that range.
    if image_np.max() > 1:
        image_np = image_np / 255.0

    # Process the attention weights.
    # Convert torch.Tensor to np.ndarray if needed.
    if isinstance(attn_weights, _torch.Tensor):
        attn_np = attn_weights.cpu().detach().numpy()
    else:
        attn_np = attn_weights

    # Normalize the attention weights to [0, 1].
    attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

    # Determine the target image dimensions.
    # image_np is in shape [W, H, C]. OpenCV resize expects (width, height) tuple.
    image_width, image_height = image_np.shape[0], image_np.shape[1]

    # Resize the attention map to match the image dimensions.
    attn_resized = _cv2.resize(attn_np, (image_height, image_width))

    # Apply a colormap to the resized attention map.
    colormap = _plt.get_cmap(cmap)
    # colormap returns an RGBA image; we take only the RGB channels.
    heatmap = colormap(attn_resized)[..., :3]

    # Overlay the heatmap on the original image.
    overlay = (1 - alpha) * image_np + alpha * heatmap
    overlay = _np.clip(overlay, 0, 1)

    # Plot the original image and the overlay.
    _plt.figure(figsize=(12, 6))

    # Original image.
    _plt.subplot(1, 2, 1)
    _plt.imshow(image_np)
    _plt.title("Original Image")
    _plt.axis('off')

    # Overlay image.
    _plt.subplot(1, 2, 2)
    _plt.imshow(overlay)
    _plt.title("Image with Attention Overlay")
    _plt.axis('off')

    _plt.tight_layout()
    _plt.show()


def display_image_with_attention_focus(
        image_tensor: _torch.Tensor,
        attn_weights: _torch.Tensor,
        min_factor: float = 0.5,
        max_factor: float = 1.5,
) -> None:
    """
    Display an image with attention-enhanced focus.

    :param image_tensor: The input image tensor.
    :param attn_weights: The attention weights tensor.
    :param min_factor: Minimum brightness multiplier.
    :param max_factor: Maximum brightness multiplier.
    """
    # Convert image tensor to NumPy array with shape [W, H, C]
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)

    # Normalize image to [0, 1] if needed
    if image_np.max() > 1:
        image_np = image_np / 255.0

    # Process the attention map: if it is a torch tensor, convert it
    if isinstance(attn_weights, _torch.Tensor):
        attn_np = attn_weights.cpu().detach().numpy()
    else:
        attn_np = attn_weights

    # Normalize the attention weights to [0, 1]
    attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

    # Resize the attention map to match the spatial dimensions of the image.
    # image_np shape is [W, H, C]. OpenCV expects size as (width, height).
    image_width, image_height = image_np.shape[0], image_np.shape[1]
    attn_resized = _cv2.resize(attn_norm, (image_height, image_width))

    # Create a brightness multiplier map.
    # For each pixel, the multiplier is min_factor (for attn=0) to max_factor (for attn=1).
    brightness_map = min_factor + (max_factor - min_factor) * attn_resized

    # Apply the brightness map on the image.
    # Make sure to apply the multiplier per channel.
    highlighted = image_np * brightness_map[..., _np.newaxis]

    # Clip the results to the [0,1] range
    highlighted = _np.clip(highlighted, 0, 1)

    # Plot the original and highlighted images side by side.
    _plt.figure(figsize=(12, 6))

    # Plot original image.
    _plt.subplot(1, 2, 1)
    _plt.imshow(image_np)
    _plt.title("Original Image")
    _plt.axis('off')

    # Plot highlighted image.
    _plt.subplot(1, 2, 2)
    _plt.imshow(highlighted)
    _plt.title("Attention-Enhanced Image")
    _plt.axis('off')

    _plt.tight_layout()
    _plt.show()
