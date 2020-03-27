import retro
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds

# generate environment
env = retro.make(game='SonicTheHedgehog-Genesis')
print('状態空間：', env.observation_space)
print('行動空間：', env.action_space)

# シードの指定
env.seed(0)
set_global_seeds(0)

# ベクトル環境の生成
env = DummyVecEnv([lambda: env])

# generate model
model = PPO2(policy=CnnPolicy, env=env, verbose=1)

# learn model
print('train...')
model.learn(total_timesteps=12800)

# model test
state = env.reset()
while True:
    env.render()
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    if done:
        env.reset()