import gym

# const num
NUM_EPISODES = 10
MAX_STEPS = 500

env = gym.make("MountainCar-v0")

for episode in range(NUM_EPISODES):
    # 環境のリセット
    env.reset()
    
    # 行動の取得
    action = env.action_space.sample()

    # 1ステップ実行
    state, reward, done, info = env.step(action)
    
    # エピソード完了時
    if done:
        print("episode:{} step:{}".format(episode, step+1))
        break