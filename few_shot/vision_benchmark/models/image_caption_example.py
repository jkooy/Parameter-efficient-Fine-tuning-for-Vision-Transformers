from torch import nn


class Example(nn.Module):
    def get_caption():
        """
        This method is called to get image caption for evaluation.
        """
        pass


def get_image_caption_model(config, **kwargs):
    """
    Specify your model here
    """
    model = Example()
    return model
