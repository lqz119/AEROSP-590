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
    self.space_dimension = 20   # 1D length in space( assume the asteroid is centered at self.space_dimension+self.asteroid_radius)
    self.dv = 0.1     # 0.1 m/s per impulse
    self.dt = 0.1       # 0.1s per time step
    self.observation_space = spaces.Discrete(int(self.space_dimension//self.dt))     # buggy here
    self.asteroid_radius = 0.5    


  def reset(self):
    # Reset the state of the environment to an initial state
    self.satellit_pos = random.randint(0, 0.1*self.space_dimension)    # random initial position 
    self.vel = random.uniform(0, self.dv*10)  # random inital velocity 
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

    #self.state = (self.satellit_pos, self.vel, self.time_spent)          # continous space
    self.state = (round(self.satellit_pos,1), round(self.vel,1), round(self.time_spent,1))  # discrete space

    terminated = bool(
            self.satellit_pos <= 0
            or self.satellit_pos >= self.space_dimension                                  # asteroid has 1 meter radius
            or abs(self.vel) >= 3 * self.asteroid_radius                                  # velocity limit to 3 times of asteroid radius
            or self.space_dimension - self.asteroid_radius * 3 <= self.satellit_pos       # success!
        )

    if terminated:
      if self.space_dimension - self.asteroid_radius * 3 <= self.satellit_pos:
        reward = 0
      else:
        reward = -10000.0
    else:
      reward = -1 * self.dt - self.dv * abs(action) 

    return np.array(self.state, dtype=np.float16), reward, terminated, False, {}






