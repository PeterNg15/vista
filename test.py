import vista

world = vista.World(["a"], trace_config={'road_width': 4})
car = world.spawn_agent(config={'length': 5.,
                                'width': 2.,
                                'wheel_base': 2.78,
                                'steering_ratio': 14.7,
                                'lookahead_road': True})
display = vista.Display(world)