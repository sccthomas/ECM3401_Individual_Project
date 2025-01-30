import cv2 as _cv2
import matplotlib.pyplot as _plt
import numpy as _np
import torch as _torch
from PIL import Image as _Image


def display_tensor_mask(mask: _torch.Tensor) -> None:
    """
    Display the mask tensor as an image.

    :param mask: The mask tensor.
    """
    mask = mask.detach().cpu().squeeze(0).numpy()
    mask = (mask * 255).astype(_np.uint8)
    mask = _Image.fromarray(mask, mode='L')
    mask.show()


def display_tensor_image(image: _torch.Tensor) -> None:
    """
    Display the image tensor as an image.

    :param image: The image tensor.
    """
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(_np.uint8)
    image = _Image.fromarray(image)
    image.show()


def display_overlaid_avg_attention(
        weights: _torch.Tensor, images: _torch.Tensor, image_idx: int = 0, alpha: float = 0.5
) -> None:
    """
    Overlay the averaged attention map on the image.

    :param weights: Attention weights of shape [num_heads, H, W] at a particular layer.
    :param images: Image tensor of shape [B, C, H, W].
    :param image_idx: Index of the image to overlay the attention map on.
    :param alpha: Weight of the overlaid attention map.
    """
    # Select the image
    img = images[image_idx].permute(1, 2, 0).detach().cpu().numpy()  # [C, H, W] -> [H, W, C]
    H, W, _ = img.shape

    # Average the attention weights over all heads
    avg_attn_map = weights.mean(dim=0).detach().cpu().numpy()

    # Resize the averaged attention map to match image size (HxW)
    avg_attn_map_resized = _cv2.resize(avg_attn_map, (H, W), interpolation=_cv2.INTER_LINEAR)

    # Normalize the attention map for better visualization
    avg_attn_map_resized = (avg_attn_map_resized - avg_attn_map_resized.min()) / (
            avg_attn_map_resized.max() - avg_attn_map_resized.min())

    # Convert the averaged attention map to a heatmap
    heatmap = _cv2.applyColorMap(_np.uint8(255 * avg_attn_map_resized), _cv2.COLORMAP_JET)

    # Overlay the heatmap on the image
    overlay = _cv2.addWeighted(heatmap, alpha, (img * 255).astype(_np.uint8), 1 - alpha, 0)

    # Display the result
    _plt.figure(figsize=(10, 5))
    _plt.subplot(1, 3, 1)
    _plt.imshow(img)
    _plt.title("Original Image")
    _plt.axis("off")

    _plt.subplot(1, 3, 2)
    _plt.imshow(avg_attn_map_resized, cmap='jet')
    _plt.title("Averaged Attention Map")
    _plt.axis("off")

    _plt.subplot(1, 3, 3)
    _plt.imshow(overlay)
    _plt.title("Overlaid Attention (Averaged)")
    _plt.axis("off")

    _plt.show()
