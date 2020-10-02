import json
import time

import numpy as np
#import pytest

import habitat
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.tasks.eqa.eqa import AnswerAction
from habitat.tasks.nav.nav import MoveForwardAction
from habitat.utils.test_utils import sample_non_stop_action

import visualpriors
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

class myDistanceToGoal():
	def __init__(self, sim, config):
		self._previous_position = None
		self._sim = sim
		self._config = config
		self._episode_view_points = None

		self.goal_radius = self._config.TASK.SUCCESS.SUCCESS_DISTANCE
		self._follower = ShortestPathFollower(self._sim, self.goal_radius, False)

	# called when a new episode start
	def reset_metric(self, episode):
		self._previous_position = None
		self._metric = None
		self._episode_view_points = [
			view_point.agent_state.position 
			for goal in episode.goals
			for view_point in goal.view_points 
			]
		self.update_metric(episode)

	def update_metric(self, episode):
		current_position = self._sim.get_agent_state().position
		distance_to_target = self.my_geodesic_distance(current_position, self._episode_view_points, episode)
		self._previous_position = current_position
		self._metric = distance_to_target

	# adapted from geodesic_distance() in habitat_simulator.py
	def my_geodesic_distance(self, position_a, position_b, episode):
		path = episode._shortest_path_cache
		path.requested_start = np.array(position_a, dtype=np.float32)

		self._sim._sim.pathfinder.find_path(path)
		return path.geodesic_distance

	# a target object has multiple viewpoints. find the closest one.
	def get_nearest_goal_viewpoint(self, episode):
		current_position = self._sim.get_agent_state().position
		path = episode._shortest_path_cache
		path.requested_start = np.array(current_position, dtype=np.float32)

		self._sim._sim.pathfinder.find_path(path)
		return path.points[-1]	

	# run the shortest_path_follower to generate the next action to go to the nearest viewpoint
	def get_next_action(self, episode):
		goal_viewpoint = self.get_nearest_goal_viewpoint(episode)
		best_action = self._follower.get_next_action(goal_viewpoint)
		return best_action
	


CFG_TEST = "configs/tasks/my_objectnav_mp3d_train.yaml"
EPISODES_LIMIT = 6
PARTIAL_LOAD_SCENES = 3

#'''
config = get_config(CFG_TEST)

dataset = make_dataset(
	id_dataset=config.DATASET.TYPE, config=config.DATASET
)
env = habitat.Env(config=config, dataset=dataset)
#goal_radius = env.episodes[0].goals[0].radius
#follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)

dg = myDistanceToGoal(env.sim, config)


for i in range(1000):
#for i in range(len(env.episodes)):
	print('******************************************************************')
	print('i == {}'.format(i))
	observations = env.reset()
	print('target_goal = {}'.format(observations['objectgoal'][0]))
	
	#assert 1==2
	action_i = 0
	len_shortest_path = len(env._current_episode.shortest_paths[0])

	# start a new episode
	dg.reset_metric(env._current_episode)

	action_list = []
	#while not env.episode_over:
	#while action_i < len_shortest_path - 1: # last action is None
	while True:
		'''
		while True:
			action = env.action_space.sample()
			if action['action'] != 'STOP':
				break


		habitat.logger.info(
			f"Action : "
			f"{action['action']}, "
			f"args: {action['action_args']}."
		)
		'''
		#action = env._current_episode.shortest_paths[0][action_i].action
		#print('action = {}'.format(action))
		predicted_action = dg.get_next_action(env._current_episode)
		#print('predicted_action = {}'.format(predicted_action))
		#action_list.append(action)
		action_list.append(predicted_action)


		#observations = env.step(action)
		observations = env.step(predicted_action)
		

		# update the position
		#'''
		dg.update_metric(env._current_episode)
		my_dist = dg._metric
		#print('my_dist = {}'.format(my_dist))
		goal_viewpoint = dg.get_nearest_goal_viewpoint(env._current_episode)
		current_position = env.sim.get_agent_state().position
		#print('current_position {}'.format(current_position))
		#print('goal_viewpoint = {}'.format(goal_viewpoint))
		#'''

		action_i += 1

		metrics = env.get_metrics()
		logger.info(metrics)
		#print('-------------------------------------------------------------------------------------------')
		if predicted_action == 0 or len(action_list) > 100:
			print('my_dist = {}'.format(my_dist))
			break
#'''

