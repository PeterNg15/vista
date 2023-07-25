import argparse
import numpy as np
import os
import cv2

import vista
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature

def follow_human_trajectory(agent):
    action = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action

# trace_paths = ["vista_traces/2021.08.17.18.54.02_veh-45_00665_01065_lidar/"]
trace_paths = ["vista_traces/20210909-160119_lexus_mit_rain_lidarevent_A2/"]

world = vista.World(
    trace_paths,
    trace_config={
        'road_width': 4,
        'master_sensor': 'lidar_3d',
    },
)

car = world.spawn_agent(
    config={
        'length': 4.084,
        'width': 1.730,
        'wheel_base': 2.58800,
        'steering_ratio': 15.2,
        'lookahead_road': True
})
lidar_config = {
    'yaw_fov': (-180., 180.),
    'pitch_fov': (-21.0, 14.0)
}
lidar = car.spawn_lidar(lidar_config)
display = vista.Display(world)

world.reset()
display.reset()

while not car.done:
    action = follow_human_trajectory(car)
    # car.step_dynamics(action)
    car.step_dataset()
    car.step_sensors()

    vis_img = display.render()
    cv2.imshow('Visualize LiDAR', vis_img[:, :, ::-1])
    cv2.waitKey(20)
