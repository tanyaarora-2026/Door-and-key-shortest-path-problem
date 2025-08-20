from utils import *
from create_env import DoorKey10x10Env
from utils import *
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
import itertools as it
import minigrid
MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


############Defining State Space#############
def define_state_space(env, info, random_env=False):
    """
    Generalized state space definition for both Part A and Part B.

    Args:
        env      : Gym MiniGrid environment
        info     : Dictionary with env metadata
        random_env: Set True for Part B (unknown key/goal positions)

    Returns:
        state_index, state_space, goal, key_pos, door_pos, empty, num_doors
    """
    ######### of from here#########
    # Positions in grid
    cell = list(range(env.height))
    positions = tuple(it.product(cell, repeat=2))
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    key_vals = [0, 1]

    # Door State
    if random_env:
        # Part B → 2 doors, each can be open/closed
        num_doors = 2
        door_vals = list(it.product([0, 1], repeat=num_doors))  # [(0,0), (0,1), (1,0), (1,1)]
        door_pos = [(5, 3), (5, 7)]  # Hardcoded as per spec
    else:
        # Part A → 1 door, either open (1) or closed (0)
        num_doors = 1
        door_vals = [0, 1]
        door_pos = tuple(info['door_pos'])  # still use env info here

    # States
    states = {
        'agent_pos': positions,
        'agent_dir': directions,
        'key': key_vals,
        'door': door_vals
    }

    if random_env:
        key_pos_list = [(1, 1), (2, 3), (1, 6)]
        goal_list    = [(5, 1), (6, 3), (5, 6)]
        states['key_pos'] = key_pos_list
        states['goal'] = goal_list
        key_pos = None
        goal = None
    else:
        key_pos = tuple(info['key_pos'])
        goal = tuple(info['goal_pos'])

    # Create state space
    keys, values = zip(*states.items())
    state_space = [dict(zip(keys, x)) for x in it.product(*values)]
    print(f'[INFO] Total number of states: {len(state_space)}')

    # Create index mapping
    state_index = {}
    for i, s in enumerate(state_space):
        state_index[tuple(s.items())] = i

    # env_matrix = gym_minigrid.minigrid.Grid.encode(env.grid)[:, :, 0]
    env_matrix = minigrid.core.grid.Grid.encode(env.grid)[:, :, 0]  # 0th channel = object type
    empty = np.where(env_matrix == 1)
    empty = tuple(zip(empty[0], empty[1]))

    return state_index, state_space, goal, key_pos, door_pos, empty, num_doors
    ######### to here##############

def example_use_of_gym_env():
    """
    The Coordinate System:
        (0,0): Top Left Corner
        (x,y): x-th column and y-th row
    """

    print("<========== Example Usages ===========> ")
    env_path = "D:/ece276b/ECE276B_PR1/starter_code/envs/known_envs/doorkey-8x8-direct.env"
    env, info = load_env(env_path) # load an environment

    # env, info = load_env("./envs/known_envs/doorkey-8x8-shortcut.env")
    print("<Environment Info>\n")
    print(info)  # Map size
    # agent initial position & direction,
    # key position, door position, goal position
    print("<================>\n")

    state_index, state_space, goal, key_pos, door_pos, empty, num_doors = define_state_space(env, info, random_env=False)

    print(empty)
    print(np.shape(state_space))
    print("1st 3 states")

    print(state_space[:3])
    print("Key_pos", key_pos)
    print("door_pos", door_pos)
    # Visualize the environment
    plot_env(env)



    # Get the agent position
    agent_pos = env.agent_pos

    # Get the agent direction
    agent_dir = env.dir_vec  # or env.agent_dir

    # Get the cell in front of the agent
    front_cell = env.front_pos  # == agent_pos + agent_dir

    # Access the cell at coord: (2,3)
    cell = env.grid.get(2, 3)  # NoneType, Wall, Key, Goal

    # Get the door status
    door = env.grid.get(info["door_pos"][0], info["door_pos"][1])
    is_open = door.is_open
    is_locked = door.is_locked

    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None

    # Take actions
    cost, done = step(env, MF)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Moving Forward Costs: {}".format(cost))
    cost, done = step(env, TL)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Turning Left Costs: {}".format(cost))
    cost, done = step(env, TR)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Turning Right Costs: {}".format(cost))
    cost, done = step(env, PK)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Picking Up Key Costs: {}".format(cost))
    cost, done = step(env, UD)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Unlocking Door Costs: {}".format(cost))

    # Determine whether we stepped into the goal
    if done:
        print("Reached Goal")

    # The number of steps so far
    print("Step Count: {}".format(env.step_count))
    

example_use_of_gym_env()


    # # Door states
    # if 'door_open' in info:
    #     num_doors = len(info['door_open'])
    #     from itertools import product
    #     door_vals = list(product([0, 1], repeat=num_doors))
    #     door_pos = [tuple(p) for p in info['door_pos']]
    # else:
    #     num_doors = 1
    #     door_vals = [0, 1]
    #     door_pos = tuple(info['door_pos'])
    
    # # For Part A: extract fixed key_pos, goal
    # if not random_env:
    #     key_pos = tuple(info['key_pos'])
    #     goal = tuple(info['goal_pos'])
    # else:
    #     key_pos = None
    #     goal = None

    # Extract empty (walkable) cells
    
