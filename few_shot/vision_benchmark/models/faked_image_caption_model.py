'''
This module defines a faked image caption model. It simuates the process of image caption
by using the first sentence in the ground truth file.
'''
from torch import nn


class FakedImageCaptionModel(nn.Module):
    def __init__(self, config):
        super(FakedImageCaptionModel, self).__init__()

    def get_caption(self, image, labels):
        """
        This method is called to get image caption for evaluation.
        """
        # Get faked image caption by using the first sentence in the labels
        res = [label[0] if label else '' for label in labels]
        return res


def load_faked_image_caption_model(config, **kwargs):
    """
    Specify your model here
    """
    model = FakedImageCaptionModel(config)
    return model
