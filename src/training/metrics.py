import torch
import torch as _torch
import torchmetrics.segmentation as _metrics


class SegmentationMetrics:
    """
    Class to handle all metric calculation during model training.
    """

    def __init__(self, len_dataset: int) -> None:
        """

        :param len_dataset: The length of the training or validation dataset.
        """
        # Metric Calculators
        num_classes = 1
        device = torch.device('cpu')
        self.__dice_score = _metrics.DiceScore(
            average='micro',
            num_classes=num_classes,
        ).to(device)
        self.__generalized_dice_score = _metrics.GeneralizedDiceScore(num_classes=num_classes).to(device)
        self.__hausdorff_distance = _metrics.HausdorffDistance(
            distance_metric='euclidean',
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
        self.__generalized_dice_score_metric = 0
        self.__hausdorff_distance_metric = 0
        self.__mean_intersection_over_union_metric = 0

    @property
    def binary_cross_entropy_loss(self) -> float:
        """
        Get the Binary Cross Entropy Loss.
        :return: The Binary Cross Entropy Loss.
        """
        return self.__binary_cross_entropy_loss

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
        generalized_dice_score = self.__generalized_dice_score
        hausdorff_distance = self.__hausdorff_distance
        mean_intersection_over_union = self.__mean_intersection_over_union
        # Metric Values
        dice_score_metric = self.__dice_score_metric
        generalized_dice_score_metric = self.__generalized_dice_score_metric
        hausdorff_distance_metric = self.__hausdorff_distance_metric
        mean_intersection_over_union_metric = self.__mean_intersection_over_union_metric

        preds = (_torch.sigmoid(preds) > 0.5).detach().to(device).int()
        target = target.detach().to(device).int()
        dice_score_metric += dice_score(preds, target).item()
        generalized_dice_score_metric += generalized_dice_score(preds, target).item()
        hausdorff_distance_metric += hausdorff_distance(preds, target).item()
        mean_intersection_over_union_metric += mean_intersection_over_union(preds, target).item()

        self.__binary_cross_entropy_loss += binary_cross_entropy_loss
        self.__dice_score_metric = dice_score_metric
        self.__generalized_dice_score_metric = generalized_dice_score_metric
        self.__hausdorff_distance_metric = hausdorff_distance_metric
        self.__mean_intersection_over_union_metric = mean_intersection_over_union_metric

    def end_of_epoch(self) -> None:
        """
        End of epoch so divide all total loss values to get average loss for epoch.
        """
        len_dataset = self.__len_dataset

        try:
            self.__binary_cross_entropy_loss /= len_dataset
            self.__dice_score_metric /= len_dataset
            self.__generalized_dice_score_metric /= len_dataset
            self.__hausdorff_distance_metric /= len_dataset
            self.__mean_intersection_over_union_metric /= len_dataset
        except ZeroDivisionError:
            print("There are no values to average")

    def reset_metrics(self) -> None:
        """
        Reset all metrics.
        """
        self.__binary_cross_entropy_loss = 0
        self.__dice_score_metric = 0
        self.__generalized_dice_score_metric = 0
        self.__hausdorff_distance_metric = 0
        self.__mean_intersection_over_union_metric = 0

    def __str__(self) -> str:
        binary_cross_entropy_loss = self.__binary_cross_entropy_loss
        dice_score_metric = self.__dice_score_metric
        generalized_dice_score_metric = self.__generalized_dice_score_metric
        hausdorff_distance_metric = self.__hausdorff_distance_metric
        mean_intersection_over_union_metric = self.__mean_intersection_over_union_metric

        msg = '--------------------------------------------\n'
        msg += f'Average Binary Cross Entropy Loss: {binary_cross_entropy_loss} \n'
        msg += f'Average Dice Score: {dice_score_metric} \n'
        msg += f'Average Generalized Dice Score: {generalized_dice_score_metric} \n'
        msg += f'Average Hausdorff Distance: {hausdorff_distance_metric} \n'
        msg += f'Average Mean IoU: {mean_intersection_over_union_metric} \n'
        msg += '--------------------------------------------\n'

        return msg
