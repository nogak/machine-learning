import gym
from gym.spaces import *

# env ID
ENV_ID = 'CartPole-v1'

# 空間の出力
def print_spaces(label, space):
    print(label, space)

    # Box/Discreteの場合は最大値と最小値も表示
    if isinstance(space, Box):
        print('  最小値：', space.low)
        print('  最大値：', space.high)
    if isinstance(space, Discrete):
        print('  最小値：', 0)
        print('  最大値：', space.n-1)

# 環境の生成
env = gym.make(ENV_ID)

# 状態空間と行動空間の型の出力
print('環境 ID : ', ENV_ID)
print_spaces('状態空間 : ', env.observation_space)
print_spaces('行動空間 : ', env.action_space)