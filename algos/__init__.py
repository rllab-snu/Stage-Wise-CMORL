from algos.comoppo import Agent as COMOPPO
# from algos.p3o import Agent as P3O
# from algos.ppo import Agent as PPO
from algos.student import Agent as Student

algo_dict = {
    'comoppo': COMOPPO,
    # 'p3o': P3O,
    # 'ppo': PPO,
    'student': Student,
}