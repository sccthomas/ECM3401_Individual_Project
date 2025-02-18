import abc as _abc

import torch as _torch
import torch.nn as _nn

import src.vision_transformer.model as _model


class SelfSupervisedLoss(_nn.Module):
    """
    Base class for self-supervised learning classes that apply self supervised learning and return a loss value.
    """

    def __init__(
            self,
            model: _model.SemanticSegmentationVisionTransformer,
    ) -> None:
        """
        Initialize the contrastive pre-training mixin.

        :param model: The model to be trained.
        """
        super(SelfSupervisedLoss, self).__init__()
        self.__model = model

    @property
    def model(self) -> _model.SemanticSegmentationVisionTransformer:
        """
        Get the model being trained.

        :return: The model.
        """
        return self.__model

    @_abc.abstractmethod
    def forward_loss(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward loss pass of the self supervised pre-training.

        :param x: The input tensor.
        :return: The self-supervised loss.
        """

    @_abc.abstractmethod
    def __loss_fn(self, **kwargs) -> _torch.Tensor:
        """
        Compute the current self-supervised loss.

        :return: A loss tensor.
        """
