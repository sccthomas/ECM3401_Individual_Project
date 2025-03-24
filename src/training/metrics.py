import torch
import torch as _torch
import torchmetrics.segmentation as _metrics


class SegmentationMetrics:
    """
    Class to handle all metric calculation during model training.
    """

    def __init__(self, len_dataset: int, device: torch.device) -> None:
        """

        :param len_dataset: The length of the training or validation dataset.
        """
        # Metric Calculators
        num_classes = 1
        self.__dice_score = _metrics.DiceScore(
            average='micro',
            num_classes=num_classes,
        ).to(device)
        self.__mean_intersection_over_union = _metrics.MeanIoU(
            num_classes=num_classes,
        ).to(device)
        self.__device = device
        # Metric Values
        self.__len_dataset = len_dataset
        self.__binary_cross_entropy_loss = 0
        self.__dice_score_metric = 0
        self.__mean_intersection_over_union_metric = 0
        self.__memory = ""

    @property
    def binary_cross_entropy_loss(self) -> float:
        """
        Get the Binary Cross Entropy Loss.

        :return: The Binary Cross Entropy Loss.
        """
        return self.__binary_cross_entropy_loss

    @property
    def memory(self) -> str:
        """
        Get the memory of the metrics.

        :return: The memory of the metrics.
        """
        return self.__memory

    def update_metrics(
            self, preds: _torch.Tensor, target: _torch.Tensor, binary_cross_entropy_loss: float
    ) -> None:
        """
        Update the metrics given a prediction tensor.

        :param preds: The prediction tensor.
        :param target: The target for each prediction.
        :param binary_cross_entropy_loss: Current Binary Cross Entropy Loss.
        """
        device = self.__device
        # Metric Calculators
        dice_score = self.__dice_score
        mean_intersection_over_union = self.__mean_intersection_over_union

        preds = preds.detach().to(device)
        target = target.detach().to(device)

        preds = (_torch.sigmoid(preds) > 0.5).int()
        target = target.int()

        self.__binary_cross_entropy_loss += binary_cross_entropy_loss
        self.__dice_score_metric += dice_score(preds, target).item()
        self.__mean_intersection_over_union_metric += mean_intersection_over_union(preds, target).item()

    def end_of_epoch(self) -> None:
        """
        End of epoch so divide all total loss values to get average loss for epoch.
        """
        len_dataset = self.__len_dataset
        memory = self.__memory

        try:
            self.__binary_cross_entropy_loss /= len_dataset
            self.__dice_score_metric /= len_dataset
            self.__mean_intersection_over_union_metric /= len_dataset
        except ZeroDivisionError:
            print("There are no values to average")

        self.__memory = memory + self.__str__()

    def reset_metrics(self) -> None:
        """
        Reset all metrics.
        """
        self.__binary_cross_entropy_loss = 0
        self.__dice_score_metric = 0
        self.__mean_intersection_over_union_metric = 0

    def __str__(self) -> str:
        """
        Return the string representation of the metrics.

        :return: The string representation of the metrics.
        """
        binary_cross_entropy_loss = self.__binary_cross_entropy_loss
        dice_score_metric = self.__dice_score_metric
        mean_intersection_over_union_metric = self.__mean_intersection_over_union_metric

        msg = '--------------------------------------------\n'
        msg += f'Average Binary Cross Entropy Loss: {binary_cross_entropy_loss} \n'
        msg += f'Average Dice Score: {dice_score_metric} \n'
        msg += f'Average Mean IoU: {mean_intersection_over_union_metric} \n'
        msg += '--------------------------------------------\n'

        return msg
