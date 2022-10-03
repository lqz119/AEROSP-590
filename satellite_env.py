import gym
from gym import spaces
import numpy as np
import random

class SatelliteEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  #metadata = {'render.modes': ['human']}

  def __init__(self, ):
    super(SatelliteEnv, self).__init__()
    #self.reward_range = (-1000, 10000) 
    self.action_space = spaces.Discrete(3)  # 0 for off, 1 for right pulse, 2 for left pulse,
    self.observation_space = spaces.Discrete(3)
    self.dv = 0.1     # 0.1 m/s per impulse
    self.dt = 1       # 1s per time step
    self.asteroid_radius = 1    # 1m



  def reset(self):
    # Reset the state of the environment to an initial state
    self.satellit_pos = random.randint(1, 100)    # initial position anywhere at 1 - 100
    self.vel = random.uniform(0, 5)  # inital 1d velocity anywhere in 0 - 5
    self.time_spent = 0
    self.state = (self.satellit_pos, self.vel, self.time_spent)
    return np.array(self.state, dtype=np.float16), {}


  def step(self, action):
    if action == 1:
      self.vel += self.dv * self.dt
    elif action == 2:
      self.vel -= self.dv * self.dt

    self.time_spent += 1

    self.satellit_pos += self.vel * self.dt

    self.state = (self.satellit_pos, self.vel, self.time_spent)

    terminated = bool(
            self.satellit_pos <= 0
            or self.satellit_pos > 500 - self.asteroid_radius                                         # asteroid has 1 meter radius
            or abs(self.vel) >= 6 * self.asteroid_radius                                              # velocity limit to 3 times of asteroid diameter
            or 500 - self.asteroid_radius - 3 <= self.satellit_pos <= 500 - self.asteroid_radius      # success!
        )

    if terminated:
      if 500 - self.asteroid_radius - 3 <= self.satellit_pos <= 500 - self.asteroid_radius:
        reward = 0
      else:
        reward = -10000.0
    else:
      reward = -1 * self.dt - self.dv * abs(action) 

    return np.array(self.state, dtype=np.float16), reward, terminated, False, {}






