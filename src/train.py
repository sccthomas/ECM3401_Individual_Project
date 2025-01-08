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
        criterion: '_nn.modules.loss._Loss',
        device: _torch.device,
        num_epochs: int,
) -> _nn.Module:
    _log.info(f"Training model for {num_epochs} epochs")
    for epoch in range(num_epochs):
        # Training phase
        _log.info(f"Starting training for epoch: {epoch + 1}")
        model.train()
        training_loss = 0.0
        optimizer.zero_grad()  # Zero the gradients before starting the loop

        for step, dataset_batch in enumerate(training_dataset_loader):
            batch_images, batch_masks = dataset_batch
            batch_images, batch_masks = batch_images.to(device), batch_masks.to(device)

            # Forward pass
            predictions = model(batch_images)
            _log.info(f"Finished forward pass for epoch: {epoch + 1} | step: {step + 1}")
            loss = criterion(predictions, batch_masks)
            loss.backward()

            # Gradient accumulation (if using more than 1 step per gradient update)
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
                batch_images, batch_masks = batch_images.to(device), batch_masks.to(device)
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
