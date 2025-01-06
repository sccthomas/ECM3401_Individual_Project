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
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        training_loss = 0.0
        for dataset_batch in training_dataset_loader:
            batch_images, batch_masks = dataset_batch
            batch_images, batch_masks = batch_images.to(device), batch_masks.to(device)
            predictions = model(batch_images)
            loss = criterion(predictions, batch_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss /= len(training_dataset_loader)

        # Validation phase
        model.eval()
        validation_loss = 0.0
        with _torch.no_grad():
            for dataset_batch in validation_dataset_loader:
                batch_images, batch_masks = dataset_batch
                batch_images, batch_masks = batch_images.to(device), batch_masks.to(device)
                predictions = model(batch_images)
                loss = criterion(predictions, batch_masks)
                validation_loss += loss.item()

        validation_loss /= len(validation_dataset_loader)

        print(f"Finished Epoch: {epoch} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f}")

    return model
