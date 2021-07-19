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

LOOPS = 50
SCENE_FILE = "reach_b.ttt"
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
agent = Panda()
gripper = PandaGripper()
workspace = Shape('workspace')
camera = VisionSensor('camera')

# We could have made this target in the scene, but lets create one dynamically
target = Shape.create(type=PrimitiveShape.CUBOID,
                      size=[0.05, 0.05, 0.05],
                      color=[1.0, 1.0, 1.0],
                      static=True, respondable=False)

position_min, position_max = [-0.2, -0.2, 0], [0.2, 0.2, 0]

starting_joint_positions = agent.get_joint_positions()

position1 = [np.array([1, 1, 1])]
position = []

for i in range(LOOPS):

    # Reset the arm at the start of each 'episode'
    agent.set_joint_positions(starting_joint_positions)

    # Get a random position within a cuboid and set the target position
    pos = list(np.random.uniform(position_min, position_max))
    target.set_position(pos, relative_to=workspace)

    # Get a path to the target (rotate so z points down)
    try:
        path = agent.get_path(
            position=pos, euler=[0, math.radians(180), 0], relative_to=workspace)
    except ConfigurationPathError as e:
        print('Could not find path')
        continue

    # Step the simulation and advance the agent along the path
    done = False
    j = 0
    while not done:
        done = path.step()
        pr.step()
        current_positions = gripper.get_position(relative_to=workspace)
        # print(position[-1])
        # dis = pos - current_positions
        # dis = math.sqrt(math.pow(dis[0], 2) + math.pow(dis[1], 2) + math.pow(dis[1], 2))
        if position1[-1][2] > current_positions[2]:
            j = j + 1
            img = camera.capture_rgb()
            img = Image.fromarray(np.uint8(img * 255))
            position.append(current_positions)
            img.save('/home/nam/workspace/imitation/data/set_reach_b/img_{}_{}.png'.format(i, j))
        position1.append(current_positions)
    
    print('Reached target %d!' % i)
dataframe = pd.DataFrame(position)
dataframe.to_csv('/home/nam/workspace/imitation/data/position/set_reach_b.csv', header=False, index=False)


pr.stop()
pr.shutdown()