
from abc import ABC, abstractmethod


class TaskBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self, client_id, robot):
        """Set up task-specific objects/goals in the environment."""
        pass

    @abstractmethod
    def get_obs(self, robot):
        """Return observation for policy/training (e.g., robot state, object positions)."""
        pass

    @abstractmethod
    def reward(self, robot):
        """Return scalar reward for current step."""
        pass

    @abstractmethod
    def check_success(self, robot):
        """Return True if task is complete, False otherwise."""
        pass