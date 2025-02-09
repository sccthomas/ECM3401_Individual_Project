from contextlib import nullcontext

import torch
import torch.utils.data as _data
import tqdm as _tqdm

import src.self_supervised_learning.base as _base

# Try to import torch_xla for TPU support. If not available (e.g. when training on GPU),
# xm will be set to None.
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def train_model(
        ssl_model: _base.SelfSupervisedLoss,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scaler: torch.cuda.amp.GradScaler,
        train_loader: _data.DataLoader,
        val_loader: _data.DataLoader,
        patience: int,
        device: torch.device,
) -> None:
    """
    Train a self-supervised model on GPU or TPU.

    :param ssl_model: The self-supervised learning model (must implement forward_loss).
    :param num_epochs: Number of training epochs.
    :param optimizer: The optimizer to update model parameters.
    :param scheduler: Learning rate scheduler.
    :param scaler: Gradient scaler (for GPU mixed-precision). This is ignored for TPU.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param patience: Number of epochs with no improvement before early stopping.
    :param device: The device to use (e.g., torch.device('cuda') or xm.xla_device() for TPU).
    """
    best_val_loss = float('inf')
    patience_counter = 0
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)

    # Use mixed precision only for CUDA devices.
    if device.type == 'cuda':
        autocast_context = torch.amp.autocast(device_type=device.type)
    else:
        # On TPU (or CPU), simply use a no-op context.
        autocast_context = nullcontext()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # --- Training Loop ---
        ssl_model.train()
        train_loss = 0.0
        for images, _ in _tqdm.tqdm(train_loader, desc="Training"):
            # Move data to the target device (GPU or TPU)
            images = images.to(device, non_blocking=True)

            # Zero the gradients (set_to_none is only available for CUDA)
            if device.type == 'cuda':
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad()

            # Forward pass under autocast context (if applicable)
            with autocast_context:
                loss = ssl_model.forward_loss(images)

            # Accumulate loss for logging
            train_loss += loss.item()

            if device.type == 'cuda':
                # GPU: use the gradient scaler for mixed precision.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # TPU (or CPU): perform a regular backward pass and use XLA's optimizer step.
                loss.backward()
                xm.optimizer_step(optimizer)
                xm.mark_step()

            # Free temporary tensors
            del loss

        train_loss /= len_train_loader
        print(f"Training Loss: {train_loss:.4f}")

        # --- Validation Loop ---
        ssl_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in _tqdm.tqdm(val_loader, desc="Validation"):
                images = images.to(device, non_blocking=True)
                with autocast_context:
                    loss = ssl_model.forward_loss(images)
                val_loss += loss.item()
                del loss

        val_loss /= len_val_loader
        print(f"Validation Loss: {val_loss:.4f}")

        # Update learning rate scheduler
        scheduler.step()

        # --- Early Stopping and Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model checkpoints (only on TPU master if using TPU)
            if device.type == 'xla' and xm is not None:
                if xm.is_master_ordinal():
                    torch.save(ssl_model.state_dict(), "best_model_ssl.pth")
                    torch.save(ssl_model.model.state_dict(), "best_model.pth")
            else:
                torch.save(ssl_model.state_dict(), "best_model_ssl.pth")
                torch.save(ssl_model.model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Save an epoch checkpoint (again, only on TPU master if using TPU)
        if device.type == 'xla' and xm is not None:
            if xm.is_master_ordinal():
                torch.save(ssl_model.state_dict(), f"segmentation_model_ssl_epoch_{epoch + 1}.pth")
                torch.save(ssl_model.model.state_dict(), f"segmentation_model_epoch_{epoch + 1}.pth")
        else:
            torch.save(ssl_model.state_dict(), f"segmentation_model_ssl_epoch_{epoch + 1}.pth")
            torch.save(ssl_model.model.state_dict(), f"segmentation_model_epoch_{epoch + 1}.pth")

        # On GPU, clear the CUDA cache.
        if device.type == 'cuda':
            torch.cuda.empty_cache()
