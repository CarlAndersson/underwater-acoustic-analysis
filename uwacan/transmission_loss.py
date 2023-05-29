import numpy as np


class MlogR:
    def __init__(self, m=20):
        self.m = m

    def __call__(self, input_power, receiver, source, **kwargs):
        distance = receiver.position.distance_to(source.position)
        level = self.m * np.log10(distance)
        return 10**(level / 10) * input_power
