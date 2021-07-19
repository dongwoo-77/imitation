from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
import numpy as np
import math
from PIL import Image
import pandas as pd

LOOPS = 1
SCENE_FILE = "reach_b.ttt"
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
agent = Panda()
gripper = PandaGripper()
workspace = Shape('workspace')
camera = VisionSensor('camera')

# We could have made this target in the scene, but lets create one dynamically
target_r = Shape.create(type=PrimitiveShape.CUBOID,
                      size=[0.05, 0.05, 0.05],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False)

target_g = Shape.create(type=PrimitiveShape.CUBOID,
                      size=[0.05, 0.05, 0.05],
                      color=[0.1, 1.0, 0.1],
                      static=True, respondable=False)

target_b = Shape.create(type=PrimitiveShape.CUBOID,
                      size=[0.05, 0.05, 0.05],
                      color=[0.1, 0.1, 1.0],
                      static=True, respondable=False)

position_min, position_max = [-0.25, -0.25, 0], [0.25, 0.25, 0]

starting_joint_positions = agent.get_joint_positions()

position = []

for i in range(LOOPS):

    # Reset the arm at the start of each 'episode'
    agent.set_joint_positions(starting_joint_positions)

    # Get a random position within a cuboid and set the target position
    pos_r = list(np.random.uniform(position_min, position_max))
    pos_g = list(np.random.uniform(position_min, position_max))
    pos_b = list(np.random.uniform(position_min, position_max))
    target_r.set_position(pos_r, relative_to=workspace)
    target_g.set_position(pos_g, relative_to=workspace)
    target_b.set_position(pos_b, relative_to=workspace)

    # Get a path to the target (rotate so z points down)
    try:
        path = agent.get_path(
            position=pos_r, euler=[0, math.radians(180), 0], relative_to=workspace)
    except ConfigurationPathError as e:
        print('Could not find path')
        continue

    # Step the simulation and advance the agent along the path
    done = False
    j = 0
    while not done:
        j = j + 1
        done = path.step()
        pr.step()
        current_positions = gripper.get_position(relative_to=workspace)
        position.append(current_positions)
        img = camera.capture_rgb()
        img = Image.fromarray(np.uint8(img * 255))
        img.save('/home/nam/workspace/imitation/data/img/img_{}_{}.png'.format(i, j))
    
    try:
        path = agent.get_path(
            position=pos_g, euler=[0, math.radians(180), 0], relative_to=workspace)
    except ConfigurationPathError as e:
        print('Could not find path')
        continue

    # Step the simulation and advance the agent along the path
    done = False
    while not done:
        j = j + 1
        done = path.step()
        pr.step()
        current_positions = gripper.get_position(relative_to=workspace)
        position.append(current_positions)
        img = camera.capture_rgb()
        img = Image.fromarray(np.uint8(img * 255))
        img.save('/home/nam/workspace/imitation/data/img/img_{}_{}.png'.format(i, j))

    try:
        path = agent.get_path(
            position=pos_b, euler=[0, math.radians(180), 0], relative_to=workspace)
    except ConfigurationPathError as e:
        print('Could not find path')
        continue

    # Step the simulation and advance the agent along the path
    done = False
    while not done:
        j = j + 1
        done = path.step()
        pr.step()
        current_positions = gripper.get_position(relative_to=workspace)
        position.append(current_positions)
        img = camera.capture_rgb()
        img = Image.fromarray(np.uint8(img * 255))
        # img.save('/home/nam/workspace/imitation/data/img/img_{}_{}.png'.format(i, j))

    # dataframe = pd.DataFrame(position)
    # dataframe.to_csv('/home/nam/workspace/imitation/data/position/position.csv', header=False, index=False)
    print('Reached target %d!' % i)

pr.stop()
pr.shutdown()