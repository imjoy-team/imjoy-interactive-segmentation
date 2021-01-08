class InteractiveModel:
    """Interactive model interface"""

    def __init__(self, model_path, *args, **kwargs):
        """load and initialize the model"""
        pass

    def train_on_batch(self, X, y):
        """train the model for one iteration
        Parameters
        --------------
        X: array [batch_size, channel, width, height]
            the input image with 2 channels

        y: array [batch_size, channel, width, height]


        Returns
        ------------------
        None
        """
        raise NotImplementedError

    def predict(self, X):
        """predict the model for one input image
        Parameters
        --------------
        X: array [channel, width, height]
            the input image with 2 channels

        Returns
        ------------------
        y_predict: the predicted label image
        """
        raise NotImplementedError

    def save(self, model_path):
        """save the model
        Parameters
        --------------
        model_path: string
            the file path to the model

        Returns
        ------------------
        None
        """
        raise NotImplementedError

    def export(self, format, file_path):
        """export the model into different format
        Parameters
        --------------
        format: string
            the model format to be exported
        file_path: string
            the file path to the exported model

        Returns
        ------------------
        None
        """
        raise NotImplementedError
