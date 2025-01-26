import torch
import torch.nn as nn
import torch.utils.data as _data
import tqdm as _tqdm


def train_model(
        ssl_model: nn.Module,
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
        # - Training loop
        print(f"\n Epoch {epoch + 1}/{num_epochs}")
        ssl_model.train()
        train_loss = 0
        for images, masks in _tqdm.tqdm(train_loader, desc=f"Training"):
            images, masks = images.to(device), masks.to(device)
            # - Mixed Precision Forward Pass
            with torch.amp.autocast(device.type):
                loss = ssl_model(images)
            # - Update Metrics
            train_loss += loss.item()

            # - Scaler for Backward Pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # - Print Training Metrics
        train_loss /= len_train_loader
        print(f"Training Loss: {train_loss}")

        # - Validation Loop
        ssl_model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in _tqdm.tqdm(val_loader, desc=f"Validation"):
                images, masks = images.to(device), masks.to(device)
                # - Mixed Precision Forward Pass
                with torch.amp.autocast(device.type):
                    loss = ssl_model(images)
                # - Update Metrics
                val_loss += loss.item()
        # - Print Validation Metrics
        val_loss /= len_val_loader
        print(f"Validation Loss: {val_loss}")
        # - Learning rate scheduler
        scheduler.step()
        # - Early Stopping and Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(ssl_model.model.state_dict(), f"best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        # - Save model checkpoint every epoch
        torch.save(ssl_model.model.state_dict(), f"segmentation_model_epoch_{epoch + 1}.pth")
