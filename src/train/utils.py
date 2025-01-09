import logging as _log

import torch as _torch
import torch.nn as _nn
import torch.optim as _optim
import torch.utils.data as _data


def train_model(
        model: _nn.Module,
        training_dataset_loader: _data.DataLoader,
        validation_dataset_loader: _data.DataLoader,
        optimizer: _optim.Optimizer,
        criterion,
        device: _torch.device,
        num_epochs: int,
) -> _nn.Module:
    _log.info(f"Training model for {num_epochs} epochs")
    for epoch in range(num_epochs):
        # Training phase
        _log.info(f"Starting training for epoch: {epoch + 1}")
        model.train()
        optimizer.zero_grad()
        training_loss = 0.0
        for dataset_batch in training_dataset_loader:
            batch_images, batch_masks = dataset_batch
            batch_images, batch_masks = batch_images.to(device).contiguous(), batch_masks.to(device).contiguous()

            # Forward pass
            predictions = model(batch_images)
            loss = criterion(predictions, batch_masks)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()  # Clear gradients after the optimizer step

            training_loss += loss.item()

            # Clear unnecessary variables for memory efficiency
            del batch_images, batch_masks, predictions, loss
            _torch.cuda.empty_cache()  # Clear cache to free up memory

        training_loss /= len(training_dataset_loader)

        # Validation phase
        _log.info(f"Starting validation for epoch: {epoch + 1}")
        model.eval()
        validation_loss = 0.0
        with _torch.no_grad():  # No need to compute gradients for validation
            for dataset_batch in validation_dataset_loader:
                batch_images, batch_masks = dataset_batch
                batch_images, batch_masks = batch_images.to(device).contiguous(), batch_masks.to(device).contiguous()
                predictions = model(batch_images)
                loss = criterion(predictions, batch_masks)
                validation_loss += loss.item()

                # Clear unnecessary variables for memory efficiency
                del batch_images, batch_masks, predictions, loss
                _torch.cuda.empty_cache()

        validation_loss /= len(validation_dataset_loader)

        print(f"Finished Epoch: {epoch + 1}/{num_epochs} | "
              f"Training Loss: {training_loss:.4f} | "
              f"Validation Loss: {validation_loss:.4f}")

    return model
