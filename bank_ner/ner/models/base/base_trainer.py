import abc


class BaseTrainer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validate(self, *args, **kwargs):
        pass