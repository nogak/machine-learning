import retro

# generate environment
env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
env.reset()

for _ in range(2000):
    env.render()
    env.step(env.action_space.sample())