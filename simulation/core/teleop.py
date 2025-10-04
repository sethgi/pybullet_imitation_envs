
from abc import ABC, abstractmethod


class TeleopBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, obs):
        """
        Produce an action given the current observation.
        Could be from keyboard, joystick, VR, or another device.
        """
        pass
