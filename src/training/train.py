import torch
import torch.nn as nn
import torch.utils.data as _data
import tqdm as _tqdm

import src.training.metrics as _metrics
import src.vision_transformer.model.base as _base


def train_model(
        model: _base.SemanticSegmentationVisionTransformerBase,
        num_epochs: int,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scaler: torch.cuda.amp.GradScaler,
        train_loader: _data.DataLoader,
        val_loader: _data.DataLoader,
        patience: int,
        device: torch.device,
        metric_device: torch.device,
) -> None:
    """
    Function to train the model.

    :param model: The model to train.
    :param num_epochs: The number of epochs to train the model.
    :param criterion: The loss function to use.
    :param optimizer: The optimizer to use.
    :param scheduler: The learning rate scheduler to use.
    :param scaler: The gradient scaler to use.
    :param train_loader: The training data loader.
    :param val_loader: The validation data loader.
    :param patience: The number of epochs to wait before early stopping.
    :param device: The device to train the model on.
    :param metric_device: The device to calculate metrics on.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    train_metrics = _metrics.SegmentationMetrics(len_dataset=len(train_loader), device=metric_device)
    val_metrics = _metrics.SegmentationMetrics(len_dataset=len(val_loader), device=metric_device)
    for epoch in range(num_epochs):
        # - Training loop
        print(f"\n Epoch {epoch + 1}/{num_epochs}")
        model.train()
        for images, masks in _tqdm.tqdm(train_loader, desc=f"Training"):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            # - Mixed Precision Forward Pass
            with torch.amp.autocast(device.type):
                outputs, _ = model.forward(images)
                loss = criterion(outputs, masks)
            # - Update Metrics
            train_metrics.update_metrics(outputs, masks, loss.item())
            # - Scaler for Backward Pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # - Free temporary tensors
            del loss, outputs, images, masks
            torch.cuda.empty_cache()
        # - Print Training Metrics
        train_metrics.end_of_epoch()
        print("------- Training Metrics -------")
        print(train_metrics)

        # - Validation Loop
        model.eval()
        with torch.no_grad():
            for images, masks in _tqdm.tqdm(val_loader, desc=f"Validation"):
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                # - Mixed Precision Forward Pass
                with torch.amp.autocast(device.type):
                    outputs, _ = model.forward(images)
                    loss = criterion(outputs, masks)
                # - Update Metrics
                val_metrics.update_metrics(outputs, masks, loss.item())
                del loss, outputs, images, masks
                torch.cuda.empty_cache()

        # - Print Validation Metrics
        val_metrics.end_of_epoch()
        print("------- Validation Metrics -------")
        print(val_metrics)

        # - Learning rate scheduler
        scheduler.step()
        # - Early Stopping and Model Checkpoint
        val_loss = val_metrics.binary_cross_entropy_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        # - Save model checkpoint every epoch
        torch.save(model.state_dict(), f"segmentation_model_epoch_{epoch + 1}.pth")

        # - Reset Metrics
        train_metrics.reset_metrics()
        val_metrics.reset_metrics()
