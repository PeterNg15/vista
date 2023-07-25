import numpy as np
import os
import cv2

import vista
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature
import matplotlib.pyplot as plt

import copy
from matplotlib import cm
from shapely.geometry import box as Box
from shapely import affinity

from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging
from vista.tasks import MultiAgentBase
from vista.utils import transform

# trace_paths = ["./vista_traces/20210726-131912_lexus_devens_center_reverse/"]
# trace_paths = ["./vista_traces/2021.08.17.16.57.11_veh-08_01200_01636/"]
trace_paths = ["vista_traces/2021.08.17.17.17.01_veh-45_02314_02798"]


def follow_human_trajectory(agent):
    action = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action

def pure_pursuit_controller(agent):
    # hyperparameters
    lookahead_dist = 5.
    Kp = 3.
    dt = 1 / 30.

    # get road in ego-car coordinates
    ego_pose = agent.ego_dynamics.numpy()[:3]
    road_in_ego = np.array([
        transform.compute_relative_latlongyaw(_v[:3], ego_pose)
        for _v in agent.road
    ])

    # find (lookahead) target
    dist = np.linalg.norm(road_in_ego[:, :2], axis=1)
    dist[road_in_ego[:, 1] < 0] = 9999.  # drop road in the back
    tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
    dx, dy, dyaw = road_in_ego[tgt_idx]

    # simply follow human trajectory for speed
    speed = agent.human_speed

    # compute curvature
    arc_len = speed * dt
    curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
    curvature_bound = [
        tireangle2curvature(_v, agent.wheel_base)
        for _v in agent.ego_dynamics.steering_bound
    ]
    curvature = np.clip(curvature, *curvature_bound)

    return np.array([curvature, speed])


world = vista.World(trace_paths, trace_config={'road_width': 4})
car = world.spawn_agent(
    config={
        'length': 5.,
        'width': 2.,
        'wheel_base': 2.78,
        'steering_ratio': 14.7,
        'lookahead_road': True
    })
display = vista.Display(world)
state_space_controller = [
    follow_human_trajectory, pure_pursuit_controller
][1]

world.reset()
display.reset()

while not car.done:
    action = state_space_controller(car)
    car.step_dynamics(action)

    vis_img = display.render()
    cv2.imshow('Visualize control', vis_img[:, :, ::-1])
    cv2.waitKey(20)