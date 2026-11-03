import math

from .. import autograd


class loss:
    def __init__(self):
        pass

    def __call__(self):
        pass


class MAELoss(loss):
    def __init__(self):
        pass

    def __call__(self, prediction, truth):

        return (truth - prediction).abs().mean()

    def backward(self):
        self.error.backward()


class CCELoss(loss):
    def __init__(self):
        pass

    def __call__(self, prediction, truth, mask=None):
        token_loss = autograd.CrossEntropyLoss(prediction).forward(prediction, truth)

        if mask is None:
            return token_loss.mean()

        mask_size = math.prod(mask.shape)
        if mask_size != prediction.shape[0]:
            raise ValueError(
                f"mask must contain one value per prediction row; "
                f"got {mask_size} values for {prediction.shape[0]} rows"
            )

        mask = mask.reshape((prediction.shape[0], 1))
        return (token_loss * mask).sum() / mask.sum()

    def backward(self):
        self.error.backward()
