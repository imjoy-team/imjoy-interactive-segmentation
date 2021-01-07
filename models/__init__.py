class InteractiveModel:
    def __init__(self, model_path):
        pass

    def train_once(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self, model_path):
        raise NotImplementedError
