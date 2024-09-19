from .go1_backflip import Env as Go1Backflip
from .go1_sideflip import Env as Go1Sideflip
from .go1_sideroll import Env as Go1Sideroll
from .go1_twohand import Env as Go1Twohand
from .h1_backflip import Env as H1Backflip
from .h1_twohand import Env as H1Twohand

task_dict = {
    'Go1Backflip': Go1Backflip,
    'Go1Sideflip': Go1Sideflip,
    'Go1Sideroll': Go1Sideroll,
    'Go1Twohand': Go1Twohand,
    'H1Backflip': H1Backflip,
    'H1Twohand': H1Twohand
}