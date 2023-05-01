import os
import pickle
import numpy as np

example_path_mode = '/home/ylc/ARM/yt_data/sim_demo/take_target_off_desk/variation0/episodes/episode0'
example_path2 ='/home/ylc/ARM/yt_data/real_demo/2/take_target_off_desk/variation0/episodes/episode0'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'

with open(os.path.join(example_path2, LOW_DIM_PICKLE), 'rb') as c:
    datac = pickle.load(c)

# [ 0.49536437 -0.01116147  0.13662954]
with open(os.path.join(example_path_mode, LOW_DIM_PICKLE), 'rb') as f:
    data = pickle.load(f)
print(data)