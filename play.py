

## Most of this follows closely to TensorKart by Kevin Hughes
from utils import resize_image, XboxController
from termcolor import cprint

import gym
from train import create_model
import numpy as np


# Play

class Actor(object):

	def __init__(self):
		#Load in the model from train.py and the weights
		self.model = create_model(keep_prob=1) 
		self.model.load_weights('model_weights.h5')
		
		# Initialize the controller for manual override
		self.real_controller = XboxController()
		
	def get_action(self, obs):
	
		### Determine manual override
		manual_override = self.real_controller.Select == 1
		# ZACH! This is our version and will use select for manual override
		
		if not manual_override:
			# look
			vec = resize_image(obs)
			vec = np.expand_dims(vec, axis=0) # Expand dimensions for predict
			# Think
			joystick = self.model.predict(vec, batch_size = 1)[0]
			
		else:
			joystick = self.real_controller.read()
			joystick[1] *= -1 # flip y (this in in the real config when it runs normally)
			
		#Act
		
		# Calibrations
		# Zach, here are the outputs you'll have to change for our simulator
		output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
            int(round(joystick[2])),
            int(round(joystick[3])),
            int(round(joystick[4])),
        ]
		
		
		# print to console
		if manual_override:
			cprint("Manual: " + str(output), 'yellow')
		else:
			cprint("AI: " + str(output), 'green')
			
		return output
		
if __name__ == '__main__':

	env = gym.make('Mario-Kart-Royal-Raceway-v0')
	# This will have to be different for ours Zach, but idk how
	obs = env.reset()
	env.render()
	print('env ready!')
	
	actor = Actor()
	print('Actor ready!')
	
	print('beginning episode loop')
	total_reward = 0
	end_episode = False
	while not end_episode:
		action - actor.get_action(obs)
		obs, reward, end_episode, info = env.step(action)
		env.render()
		total_reward += reward
	print('end episode... total reward: ' + str(total_reward))
	
	obs = env.reset()
	print('env ready!')
	
	input('press <enter> to quit')
	
	env.close()
	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		