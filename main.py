import datetime as _datetime
import logging as _log
import os as _os

import torch as _torch
import torch.nn as _nn
import torch.optim as _optim
import torch.utils.data as _data

import dataset.snow as _snow
import model.config as _config
import model.model as _model


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


def config_1() -> _nn.Module:
    # Create model
    model_config = _config.ModelConfig.create(
        input_dimensions=(3, 512, 512),
        output_dimensions=(1, 512, 512),
        num_encoder_stages=2,
        num_classes=1,
        patch_embedding_config_dicts=[
            # Patch Embedding Config 1
            {
                "patch_embedding_info": {
                    'patch_size': 128,
                    'in_channels': 1024,
                },
                "encoder_block_configs": [
                    {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (2, 2)
                    }
                    for _ in range(3)
                ],
                "decoder_block_config": {
                    'dropout': True,
                    'iterations': 3,
                    'num_attention_heads': 8,
                    'shifted_window': True,
                    'window_size': (2, 2)
                }
            },
            # Patch Embedding Config 2
            {
                "patch_embedding_info": {
                    'patch_size': 64,
                    'in_channels': 768,
                },
                "encoder_block_configs": [
                    {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (4, 4)
                    }
                    for _ in range(3)
                ],
                "decoder_block_config": {
                    'dropout': True,
                    'iterations': 3,
                    'num_attention_heads': 8,
                    'shifted_window': True,
                    'window_size': (4, 4)
                }
            },
            # Patch Embedding Config 3
            {
                "patch_embedding_info": {
                    'patch_size': 32,
                    'in_channels': 512,
                },
                "encoder_block_configs": [
                    {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (4, 4)
                    }
                    for _ in range(3)
                ],
                "decoder_block_config": {
                    'dropout': True,
                    'iterations': 3,
                    'num_attention_heads': 8,
                    'shifted_window': True,
                    'window_size': (4, 4)
                }
            },
            # Patch Embedding Config 4
            {
                "patch_embedding_info": {
                    'patch_size': 16,
                    'in_channels': 256,
                },
                "encoder_block_configs": [
                    {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (8, 8)
                    }
                    for _ in range(3)
                ],
                "decoder_block_config": {
                    'dropout': True,
                    'iterations': 3,
                    'num_attention_heads': 8,
                    'shifted_window': True,
                    'window_size': (8, 8)
                }
            },
            # Patch Embedding Config 5
            {
                "patch_embedding_info": {
                    'patch_size': 8,
                    'in_channels': 64,
                },
                "encoder_block_configs": [
                    {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (16, 16)
                    }
                    for _ in range(3)
                ],
                "decoder_block_config": {
                    'dropout': True,
                    'iterations': 3,
                    'num_attention_heads': 8,
                    'shifted_window': True,
                    'window_size': (16, 16)
                }
            },
        ],
    )
    device = _torch.device('cuda')  # Fallback to CUDA if MPS is not available
    semantic_segmentation_model = _model.SemanticSegmentationVisionTransformer.from_config(model_config).to(device)

    # Create dataset loaders
    snow_dataset = _snow.SnowDataset(dataset_dir_path='/content/drive/MyDrive/snow_dataset')
    training_dataset, validation_dataset = _data.random_split(snow_dataset, [0.8, 0.2])

    batch_size = 1
    training_dataset_loader = _data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset_loader = _data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Create Optimizer, Loss function and Device
    optimizer = _optim.Adam(semantic_segmentation_model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = _nn.BCEWithLogitsLoss().to(device)

    # Train model
    num_epochs = 1
    semantic_segmentation_model = train_model(
        semantic_segmentation_model,
        training_dataset_loader,
        validation_dataset_loader,
        optimizer,
        criterion,
        device,
        num_epochs,
    )

    # Save model
    data_str = _datetime.datetime.now().strftime('%Y-%m-%d')
    folder_path = f'trained_model/'
    _os.makedirs(folder_path, exist_ok=True)
    _torch.save(semantic_segmentation_model.state_dict(), f'{folder_path}{data_str}_semantic_segmentation_model.pth')

    return semantic_segmentation_model
