import itertools as it
import minigrid
from utils import *
from tqdm import tqdm
from create_env import DoorKey10x10Env

####### Control Inputs ########### 
MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

forward_rotation = {}
forward_rotation[(0,-1)] = {TL : (-1,0), TR : (1,0)}
forward_rotation[(0,1)] = {TL : (1,0), TR : (-1,0)}
forward_rotation[(-1,0)] = {TL : (0,1), TR : (0,-1)}
forward_rotation[(1,0)] = {TL : (0,-1), TR : (0,1)}

############ Defining State Space #############
def define_state_space(env, info, random_env=True):
    """
    Generalized state space definition for both Part A and Part B.

    Args:
        env      : Gym MiniGrid environment
        info     : Dictionary with env metadata
        random_env: Set True for Part B (unknown key/goal positions)

    Returns:
        state_index: Mapping from state → index
        state_space: List of all state dictionaries
        goal: Tuple of goal position
        key_pos: Tuple of key position 
        door_pos: List of door positions
        empty: List of empty (x, y) positions
        num_doors: Int (1 or 2)

    """
    # Positions in grid
    cell = list(range(env.height))
    positions = tuple(it.product(cell, repeat=2))
    UP, RIGHT, DOWN, LEFT = (0, -1), (1, 0), (0, 1), (-1, 0)
    directions = [UP, RIGHT, DOWN, LEFT]
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
        key_pos_list = [(2, 2), (2, 3), (1, 6)]
        goal_list    = [(6, 1) ,(7,3), (6,6)]
        states['key_pos'] = key_pos_list
        states['goal'] = goal_list
        key_pos = tuple(int(v) for v in info['key_pos'])
        goal = tuple(int(v) for v in info['goal_pos'])
    else:
        key_pos = tuple(int(v) for v in info['key_pos'])
        goal = tuple(int(v) for v in info['goal_pos'])
        states['key_pos'] = [key_pos]
        states['goal']    = [goal]

    # Create state space
    keys, values = zip(*states.items())
    state_space = [dict(zip(keys, x)) for x in it.product(*values)]
    print(f'[INFO] Total number of states: {len(state_space)}')
    # Force all ints and a consistent dict order
    for s in state_space:
        s['agent_pos'] = (int(s['agent_pos'][0]), int(s['agent_pos'][1]))
        s['agent_dir'] = (int(s['agent_dir'][0]), int(s['agent_dir'][1]))
        s['key'] = int(s['key'])
        if isinstance(s['door'], (tuple, list)):
            s['door'] = tuple(int(x) for x in s['door'])
        else:
            s['door'] = int(s['door'])
        s['goal'] = (int(s['goal'][0]), int(s['goal'][1]))
        s['key_pos'] = (int(s['key_pos'][0]), int(s['key_pos'][1]))

    state_index = {tuple(sorted(s.items())): i for i, s in enumerate(state_space)}

    # Empty or Walkable cells
    env_matrix = minigrid.core.grid.Grid.encode(env.grid)[:, :, 0]  # 0th channel = object type
    empty = np.where(env_matrix == 1)    # FLOOR = 1
    empty = tuple(zip(empty[0], empty[1]))

    return state_index, state_space, goal, key_pos, door_pos, empty, num_doors  
    

######## Defining Motion Model ########

def motion_model(state, action, key_pos, door_pos, env, empty):
    '''
    : Unified motion model for both Part A (single door) and Part B (multiple doors)
    : Detects Part A vs Part B based on type of `door`:
        - Part A: door is a bool
        - Part B: door is a tuple (bool, bool)
    '''
    pos = state['agent_pos']
    rot = state['agent_dir']
    key = state['key']
    door = state['door']
    goal = state['goal']

    if pos == goal:
        return state
    else:
        front = tuple([pos[0] + rot[0], pos[1] + rot[1]])

        if action == MF:
            if front in empty or front == goal:
                pos = front
            if key==1 and front == key_pos:
                pos = front
            # Part A
            if isinstance(door, int):
                if door == 1 and front == door_pos:
                    pos = front
            # Part B
            elif isinstance(door, (tuple, list)):
                if front in door_pos:
                    i = door_pos.index(front)
                    if door[i]:
                        pos = front

        elif action == TL or action == TR:
            rot = forward_rotation[rot][action]

        elif action == PK:
            if not key and front == key_pos:
                key = 1

        elif action == UD:
            if key:
                if isinstance(door, int):  # Part A: single door
                    if front == door_pos and door == 0:
                        door = 1

                elif isinstance(door, (tuple, list)):  # Part B: multiple doors
                    for i, d in enumerate(door_pos):
                        if front == d and door[i] == 0:
                            # Open the i-th door by updating the tuple
                            door = tuple(1 if j == i else door[j] for j in range(len(door)))

    # Return updated state
    updated_state = {
    'agent_pos': (int(pos[0]), int(pos[1])),
    'agent_dir': (int(rot[0]), int(rot[1])),
    'key': int(key),
    'door': tuple(int(x) for x in door) if isinstance(door, (tuple, list)) else int(door),
    'goal': (int(goal[0]), int(goal[1])),
    'key_pos': key_pos
    }
   
    return updated_state


###### Backward Dynamic Programming for PartA&B with Policy Catching for PartB ######

def BackDP(state_space, state_index, controls, env, key_pos, door_pos, empty, cache_path=None, force_recompute=False):
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        print("[INFO] Loading cached policy...")
        with open(cache_path, "rb") as f:
            V, pi = pickle.load(f)
        return V, pi

    print("[INFO] Running Backward DP...")
    T = len(state_space) - 1
    # V[t, i] = cost from state i at time t to the goal
    V = np.ones((T + 1, len(state_space))) * float('inf')
    pi = np.zeros_like(V).astype(int)

    for i, s in enumerate(state_space):
        if s['agent_pos'] == s['goal']:
            V[:, i] = 0

    for t in tqdm(range(T - 1, -1, -1)):
        cij = np.zeros((len(state_space), len(controls)))
        for i, s in enumerate(state_space):
            for c, action in enumerate(controls):
                next_s = motion_model(s, action, s['key_pos'], door_pos, env, empty)
                # next_idx = state_index[tuple(next_s.items())]
                next_idx = state_index[tuple(sorted(next_s.items()))]
                cij[i, c] = step_cost(action) + V[t + 1, next_idx]

            V[t, i] = min(V[t, i], cij[i, :].min())
            pi[t, i] = controls[np.argmin(cij[i, :])]

        if all(V[t, :] == V[t + 1, :]):
            print(f"[INFO] Converged in {T - t} iterations")
            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump((V[t + 1:], pi[t + 1:]), f)
            return V[t + 1:], pi[t + 1:]

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump((V, pi), f)

    return V, pi


############# Problem Definition ##############
def doorkey_problem(env, info, random_env=True):
    controls = [MF, TL, TR, PK, UD]
    state_index, state_space, goal, key_pos, door_pos, empty, num_doors = define_state_space(env, info, random_env)

    cache_path = "D:/ece276b/ECE276B_PR1/starter_code/envs/cache/policy_partB.pkl" if random_env else None
    V, pi = BackDP(state_space, state_index, controls, env, key_pos, door_pos, empty, cache_path)

        # helper
    def door_open_from_env(env, door_pos, num_doors):
        def is_open(p):
            obj = env.grid.get(p[0], p[1])
            return int(getattr(obj, "is_open", False))
        if num_doors > 1:
            return tuple(is_open(p) for p in door_pos)
        return int(is_open(door_pos))

    # before building state:
    door_state = info.get('door_open')
    if door_state is None:
        door_state = door_open_from_env(env, door_pos, num_doors)
    else:
        door_state = tuple(int(x) for x in door_state) if num_doors > 1 else int(door_state)

    # Initialize from actual environment
    state = {
        'agent_pos': (int(env.agent_pos[0]), int(env.agent_pos[1])),
        'agent_dir': (int(env.dir_vec[0]), int(env.dir_vec[1])),  # OK for Minigrid
        'key': 0,
        'door': door_state,  # <-- no KeyError now
        'goal': (int(info['goal_pos'][0]), int(info['goal_pos'][1])),
        'key_pos': (int(info['key_pos'][0]), int(info['key_pos'][1])),
    }
    print(state)

    optimal_act_seq = []
    value_function = []
    t = 0
    while state['agent_pos'] != state['goal']:
        idx = state_index[tuple(sorted(state.items()))]
        optimal_action = pi[t, idx]
        optimal_act_seq.append(optimal_action)
        value_function.append(V[t, idx])
        state = motion_model(state, optimal_action, key_pos, door_pos, env, empty)
        t += 1

    return V, pi, optimal_act_seq, value_function, state_index, goal, key_pos, door_pos, empty, num_doors

def solution():
    
    action_dict = {0: 'MF', 1: 'TL', 2: 'TR', 3: 'PK', 4: 'UD'}   

    env_path = "D:/ece276b/ECE276B_PR1/starter_code/envs/random_envs/DoorKey-10x10-36.env"
    env, info = load_env(env_path)
    V, pi, seq, value_function, state_index, goal, key_pos, door_pos, empty, num_doors = doorkey_problem(env, info, random_env=True)
                
    action_seq = [action_dict[s] for s in seq]     
    print(f"[INFO] Sequence: {action_seq}")

    # Save GIF
    draw_gif_from_seq(seq, env, 'D:/ece276b/ECE276B_PR1/starter_code/gif/random_envs/DoorKey-10x10-36.gif')


if __name__ == '__main__':
    solution()


