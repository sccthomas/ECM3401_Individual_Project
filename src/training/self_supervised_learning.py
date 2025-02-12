import torch
import torch.utils.data as _data
import tqdm as _tqdm

import src.self_supervised_learning.base as _base


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
    Function to train the model using self-supervised learning.

    :param ssl_model: SSL API to train the model.
    :param num_epochs: The number of epochs to train the model.
    :param optimizer: The optimizer to use.
    :param scheduler: The learning rate scheduler to use.
    :param scaler: The gradient scaler to use.
    :param train_loader: The training data loader.
    :param val_loader: The validation data loader.
    :param patience: The number of epochs to wait before early stopping.
    :param device: The device to train the model on.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # --- Training Loop ---
        ssl_model.train()
        train_loss = 0.0
        for images, _ in _tqdm.tqdm(train_loader, desc="Training"):
            # - Move data to the GPU using non_blocking transfer
            images = images.to(device, non_blocking=True)
            # - Use mixed precision for forward pass
            with torch.amp.autocast(device.type):
                loss = ssl_model.forward_loss(images)
            # - Accumulate the scalar loss value (loss.item() returns a Python float)
            train_loss += loss.item()
            # - Zero gradients using set_to_none=True to reduce memory overhead
            optimizer.zero_grad(set_to_none=True)
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # - Free temporary tensors
            del loss, images
            torch.cuda.empty_cache()

        train_loss /= len_train_loader
        print(f"Training Loss: {train_loss:.4f}")

        # --- Validation Loop ---
        ssl_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in _tqdm.tqdm(val_loader, desc="Validation"):
                # - Move data to the GPU using non_blocking transfer
                images = images.to(device, non_blocking=True)
                # - Use mixed precision for forward pass
                with torch.amp.autocast(device.type):
                    loss = ssl_model.forward_loss(images)
                # - Accumulate the scalar loss value (loss.item() returns a Python float)
                val_loss += loss.item()
                # - Free temporary tensors
                del loss, images
                torch.cuda.empty_cache()

        val_loss /= len_val_loader
        print(f"Validation Loss: {val_loss:.4f}")

        # Update learning rate scheduler
        scheduler.step()

        # --- Early Stopping and Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(ssl_model.state_dict(), "best_model_ssl.pth")
            torch.save(ssl_model.model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        torch.cuda.empty_cache()
