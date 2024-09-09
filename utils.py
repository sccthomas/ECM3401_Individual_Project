from torchvision import transforms as _transforms
import os as _os
import torch as _torch


def train_model(*, model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        # Training phase
        train_loss = _train_model_for_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validation phase
        val_loss = _validate_model(model, val_loader, criterion, device)

        print(f"Finished Epoch: {epoch} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")


def save_tensor_as_image(tensor, dir, image_name) -> None:
    image = _transforms.ToPILImage()(tensor)
    filename = f'{_os.path.join(dir, image_name)}.png'
    image.save(filename)
    print('saved image to:', filename)


# ------------------------------
# Private Helpers
# ------------------------------


def _train_model_for_epoch(model, dataloader, optimizer, criterion, device, epoch) -> float:
    print("Training model | Epoch: ", epoch)
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        batch_images, batch_masks = batch
        batch_images, batch_masks = batch_images.to(device), batch_masks.to(device)
        predictions = model(batch_images)
        loss = criterion(predictions, batch_masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def _validate_model(model, dataloader, criterion, device) -> float:
    print("Validating model")
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    with _torch.no_grad():  # Disable gradient calculation for validation
        for batch in dataloader:
            batch_images, batch_masks = batch
            batch_images, batch_masks = batch_images.to(device), batch_masks.to(device)
            predictions = model(batch_images)
            loss = criterion(predictions, batch_masks)
            running_val_loss += loss.item()

    return running_val_loss / len(dataloader)
