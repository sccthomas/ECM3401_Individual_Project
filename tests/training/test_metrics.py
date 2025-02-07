import unittest

import torch

from src.training.metrics import SegmentationMetrics


class TestSegmentationMetrics(unittest.TestCase):
    def test(self) -> None:
        segmentation_metrics = SegmentationMetrics(len_dataset=10, device=torch.device('cpu'))
        preds = torch.rand(10, 1, 128, 128).to(torch.device('mps'))
        target = torch.randint(0, 2, (10, 1, 128, 128)).to(torch.device('mps'))
        segmentation_metrics.update_metrics(preds, target, 1.9)
        segmentation_metrics.end_of_epoch()

        expected_msg = '--------------------------------------------\n'
        expected_msg += f'Average Binary Cross Entropy Loss:  \n'
        expected_msg += f'Average Dice Score:  \n'
        expected_msg += f'Average Mean IoU:  \n'
        expected_msg += '--------------------------------------------\n'
        actual_msg = segmentation_metrics.__str__()
        self.assertNotEqual(expected_msg, actual_msg)

        segmentation_metrics.reset_metrics()
        expected_msg = '--------------------------------------------\n'
        expected_msg += f'Average Binary Cross Entropy Loss: 0 \n'
        expected_msg += f'Average Dice Score: 0 \n'
        expected_msg += f'Average Mean IoU: 0 \n'
        expected_msg += '--------------------------------------------\n'
        actual_msg = segmentation_metrics.__str__()
        self.assertEqual(expected_msg, actual_msg)


if __name__ == '__main__':
    unittest.main()
