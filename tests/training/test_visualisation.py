import unittest
import unittest.mock
from unittest.mock import patch, mock_open

import matplotlib.pyplot as plt
import torch

from src.training.visualisation import (
    display_tensor_mask, display_tensor_image, display_attention_weights, display_training_metrics
)


class TestDisplayFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    def test_display_tensor_mask(self) -> None:
        device = self.device

        mask = torch.rand(1, 256, 256).to(device)
        mask = (mask > 0.5).float()
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_tensor_mask(mask)
            mock_show.assert_called()

    def test_display_tensor_image(self) -> None:
        device = self.device

        image = torch.rand(3, 256, 256).to(device)
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_tensor_image(image)
            mock_show.assert_called()

    def test_display_attention_weights(self) -> None:
        device = self.device

        # Normal transformer encoder
        attention_scores = {
            ('x1', 16): {
                "encoder": [torch.rand(1, 4, 256, 256), torch.rand(1, 4, 256, 256)],
                "patch_fusion": [torch.rand(1, 4, 256, 256), torch.rand(1, 4, 256, 256)],
            }
        }

        images = torch.rand(2, 3, 256, 256).to(device)
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_attention_weights(
                image=images[0],
                attention_scores=attention_scores,
            )
            mock_show.assert_called()

        # Swin transformer encoder
        attention_scores = {
            ('x1', 16): {
                "encoder": [torch.rand(16, 4, 16, 16), torch.rand(16, 4, 16, 16)],
                "patch_fusion": [torch.rand(1, 4, 256, 256), torch.rand(1, 4, 256, 256)],
            }
        }

        images = torch.rand(2, 3, 256, 256).to(device)
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_attention_weights(
                image=images[0],
                attention_scores=attention_scores,
            )
            mock_show.assert_called()

    def test_display_training_metrics(self) -> None:
        log_data = """Length of dataset: 18209

        Epoch 1/25
        Training: 100%|██████████| 456/456 [17:08<00:00,  2.26s/it]
        ------- Training Metrics -------
        --------------------------------------------
        Average Binary Cross Entropy Loss: 0.2598952735333066
        Average Dice Score: 0.4740406656311008
        Average Mean IoU: 0.352375202170949
        --------------------------------------------

        Validation: 100%|██████████| 114/114 [02:18<00:00,  1.21s/it]
        ------- Validation Metrics -------
        --------------------------------------------
        Average Binary Cross Entropy Loss: 0.1486531374485869
        Average Dice Score: 0.7258992561122828
        Average Mean IoU: 0.5865323250753838
        --------------------------------------------

        Epoch 2/25
        Training: 100%|██████████| 456/456 [11:16<00:00,  1.48s/it]
        ------- Training Metrics -------
        --------------------------------------------
        Average Binary Cross Entropy Loss: 0.1483915554719013
        Average Dice Score: 0.7129045343189909
        Average Mean IoU: 0.5736505425812906
        --------------------------------------------

        Validation: 100%|██████████| 114/114 [00:50<00:00,  2.26it/s]
        ------- Validation Metrics -------
        --------------------------------------------
        Average Binary Cross Entropy Loss: 0.13470486754126715
        Average Dice Score: 0.7503683885984254
        Average Mean IoU: 0.6201684082809248
        --------------------------------------------
        """

        with (
            patch.object(plt, 'show') as mock_show,
            patch("builtins.open", mock_open(read_data=log_data))
        ):
            display_training_metrics("results/standard.txt")
            mock_show.assert_called()


if __name__ == '__main__':
    unittest.main()
