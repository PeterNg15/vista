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
# trace_paths = ["./vista_traces/20210726-131322_lexus_devens_center/"]
# trace_paths = ["./vista_traces/2021.08.17.16.57.11_veh-08_01200_01636/"]
# trace_paths = ["vista_traces/2021.08.17.17.17.01_veh-45_02314_02798/"]
trace_paths = ["vista_traces/2021.08.17.18.54.02_veh-45_00665_01065/"]
mesh_path = "./mesh/"

def follow_human_trajectory(agent):
    action = np.array([ 
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action

# world = vista.World(trace_paths, trace_config={'road_width': 4, 'max_timestamp_diff_across_frames':9999999999999})

world = vista.World(trace_paths, trace_config={'road_width': 4})
car = world.spawn_agent(
    config={
        'length': 4.084,
        'width': 1.730,
        'wheel_base': 2.58800,
        'steering_ratio': 15.2,
        'lookahead_road': True
    })


# car = world.spawn_agent(
#         config={
#             'length': 5.,
#             'width': 2.,
#             'wheel_base': 2.78,
#             'steering_ratio': 14.7,
#             'lookahead_road': True
#         })

# car = world.spawn_agent(
#     config={
#         'length': 4.084,
#         'width': 1.730,
#         'wheel_base': 0.8,
#         'steering_ratio': 15.2,
#         'lookahead_road': True
#     })

camera = car.spawn_camera(config={
    'size': (200, 320),
})
display = vista.Display(world)

world.reset()
display.reset()

while not car.done:
    action = follow_human_trajectory(car)
    car.step_dynamics(action, dt=1/10)
    # print(action)
    # car.step_dataset()
    car.step_sensors()

    vis_img = display.render()
    cv2.imshow('Visualize RGB', vis_img[:, :, ::-1])
    cv2.waitKey(5)

0


"""
trace_config = dict(
    road_width=4,
    reset_mode='default',
    master_sensor='camera_front',
)
car_config = dict(
    length=5.,
    width=2.,
    wheel_base=2.78,
    steering_ratio=14.7,
)
examples_path = os.path.dirname(os.path.realpath(__file__))
sensors_config = [
    dict(
        type='camera',
        # camera params
        name='camera_front',
        rig_path=os.path.join(examples_path, "params.xml"),
        size=(200, 320),
        # rendering params
        depth_mode=DepthModes.FIXED_PLANE,
        use_lighting=False,
    )
]
task_config = dict(n_agents=2,
                    mesh_dir=mesh_path,
                    init_dist_range=[6., 6.],
                    init_lat_noise_range=[0., 0.])
display_config = dict(road_buffer_size=1000, )

ego_car_config = copy.deepcopy(car_config)
ego_car_config['lookahead_road'] = True
env = MultiAgentBase(trace_paths=trace_paths,
                        trace_config=trace_config,
                        car_configs=[ego_car_config, car_config],
                        sensors_configs=[sensors_config, []],
                        task_config=task_config,
                        logging_level='DEBUG')
display = vista.Display(env.world, display_config=display_config)

# Run
env.reset()
display.reset()
done = False
while not done:
    actions = generate_human_actions(env.world)
    observations, rewards, dones, infos = env.step(actions)
    done = np.any(list(dones.values()))

    img = display.render()
    cv2.imshow("test", img[:, :, ::-1])
    cv2.waitKey(20)
"""


"""
def follow_human_trajectory(agent):
    action = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action

world = vista.World(trace_paths, trace_config={'road_width': 4})
car = world.spawn_agent(
    config={
        'length': 5.,
        'width': 2.,
        'wheel_base': 2.78,
        'steering_ratio': 14.7,
        'lookahead_road': True
    })

camera = car.spawn_camera(config={
    'size': (200, 320),
})
display = vista.Display(world)

world.reset()
display.reset()

while not car.done:
    action = follow_human_trajectory(car)
    car.step_dynamics(action)
    car.step_sensors()

    vis_img = display.render()
    cv2.imshow('Visualize RGB', vis_img[:, :, ::-1])
    cv2.waitKey(20)
"""


"""
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
"""
