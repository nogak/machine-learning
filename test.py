#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym

# generate environment
env = gym.make('MountainCar-v0')
env.reset()

# ramdom action
for i in range (2000):
    env.render()
    env.step(env.action_space.sample())