Code for running "Simultaneous Mapping and Target-driven Navigation" in Habitat simulator.

**Installation**
1. Install habitat-lab and habitat-sim from the following repository.<br />
https://github.com/facebookresearch/habitat-lab

2. The following libraries should also be necessary.<br />
torch >= 1.5<br />
torchvision == 0.6<br />
networkx<br />
visualpriors<br />

3. Put `my_objectnav_mp3d.yaml, my_objectnav_mp3d_train.yaml, my_objectnav_mp3d_test.yaml` into the *data* folder.

**Habitat Code Example**
1. `examples/shortest_path_follower_example.py`will draw a top down map visualizing the agent following a shortest path to the goal. It is also an example of running shortest path on the *pointNav* task.

2. `temp_geodesic_follower.py` is an example for running shortest path on the *objectNav* task.

**Training and Testing**

All the hyper-parameters are defined in `parameters_habitat.py`.<br />
MapNet module and navigation module are trained and tested separately.
1. To train and test the MapNet module,<br />
`python train_MapNet_habitat.py`<br />
`python test_MapNet_habitat.py`

2. To train and test the navigator,<br />
`python train_IL_habitat.py`<br />
`python test_IL_habitat.py`
