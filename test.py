import numpy as np
import os
import cv2

import vista
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature
import matplotlib.pyplot as plt


trace_paths = ["./vista_traces/20210726-131912_lexus_devens_center_reverse/"]

frames = []
cap = cv2.VideoCapture('./vista_traces/20210726-131912_lexus_devens_center_reverse/camera_front.avi')
while True:
    read, frame= cap.read()
    if not read:
        break
    frames.append(frame)


world = vista.World(trace_paths,
                    trace_config={'road_width': 4})
car = world.spawn_agent(config={'length': 5.,
                                'width': 2.,
                                'wheel_base': 2.78,
                                'steering_ratio': 14.7,
                                'lookahead_road': True})

display = vista.Display(world)

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


state_space_controller = [follow_human_trajectory, pure_pursuit_controller][1]

world.reset()
display.reset()

while not car.done:
    action = state_space_controller(car)
    car.step_dynamics(action) # update vehicle states
    car.step_sensors() # do sensor capture
    observation = car.observations # fetch sensory measurement
    car.step_dataset() # simply get next frame in the dataset without synthesis

    vis_img = display.render()[:, :, ::-1]
    frame = frames[car.frame_number]

    # diff = np.abs(vis_img.shape[1] - frame.shape[1])

    # pad_vis_img = np.concatenate([vis_img, np.zeros((vis_img.shape[1], diff, 3))], axis=1)
    bev_and_frame = np.concatenate([vis_img, frame/255.0], axis=0)

    cv2.imshow('Visualize event data', bev_and_frame)
    cv2.waitKey(1)

