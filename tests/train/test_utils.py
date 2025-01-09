import unittest

import torch
from src.model.model import SemanticSegmentationVisionTransformer
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader

from src.dataset.snow import SnowDataset
from src.train import train_model
from src.train.models.multi_scale.configs import small_model_configuration


class TestMain(unittest.TestCase):
    def test_train_model(self) -> None:
        # Create model
        device = torch.device('mps')
        semantic_segmentation_model = SemanticSegmentationVisionTransformer.from_config(
            small_model_configuration()
        ).to(device)

        # Create dataset loaders
        snow_dataset = SnowDataset(
            dataset_dir_path='/Users/samuelthomas/Documents/University/4thYr_Final'
                             '/ECM3401_Individual_Literature_Review_and_Project'
                             '/SNOW_Semantic_Segmentation'
                             '/snow_dataset'
        )
        training_dataset, validation_dataset = random_split(snow_dataset, [0.8, 0.2])

        training_dataset_loader = DataLoader(training_dataset, batch_size=10, num_workers=4, shuffle=True)
        validation_dataset_loader = DataLoader(validation_dataset, batch_size=10, num_workers=4, shuffle=False)

        # Create Optimizer, Loss function and Device
        optimizer = Adam(semantic_segmentation_model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = BCEWithLogitsLoss().to(device)

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


if __name__ == '__main__':
    unittest.main()
