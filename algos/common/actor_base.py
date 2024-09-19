from abc import ABC, abstractmethod
from typing import Tuple
import torch


@torch.jit.script
def normalize(a, minimum, maximum):
    '''
    input range: [min, max]
    output range: [-1.0, 1.0]
    '''
    temp_a = 2.0/(maximum - minimum)
    temp_b = (maximum + minimum)/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, minimum, maximum):
    '''
    input range: [-1.0, 1.0]
    output range: [min, max]
    '''
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped



class ActorBase(ABC, torch.nn.Module):
    def __init__(self, device:torch.device) -> None:
        torch.nn.Module.__init__(self)
        self.device = device

    @abstractmethod
    def build(self) -> None:
        """
        Build actor network.
        """

    @abstractmethod
    def updateActionDist(self, state:torch.Tensor, epsilon:torch.Tensor) -> None:
        """
        Return unnormalized action and normalized action.
        """

    @abstractmethod
    def sample(self, deterministic:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return unnormalized action and normalized action.
        """

    @abstractmethod
    def getDist(self) -> torch.distributions.Distribution:
        """
        Return action distribution given state.
        If state is None, use the internal state set in the `sample` function.
        """

    @abstractmethod
    def getEntropy(self) -> torch.Tensor:
        """
        Return action entropy given state.
        If state is None, use the internal state set in the `sample` function.
        """

    @abstractmethod
    def getLogProb(self) -> torch.Tensor:
        """
        Return action log probability given state and action.
        If state and action is None, use the internal state and action set in the `sample` function.
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize actor's parameters.
        """
