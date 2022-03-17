import abc


class BasePredictor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass