from humoro.trajectory import Trajectory
from humoro.load_scenes import autoload_objects
from humoro.gaze import load_gaze
import h5py
from humoro.player_pybullet import Player
from humoro.kin_pybullet import HumanKin
import humoro._math_utils as mu

import numpy as np
import pickle
from config import mykeys, mydata
import os
pp = Player()

for dataset in mydata:
    print(f"Importing: {dataset}")
    os.system("mkdir ../data/" + dataset)
    obj_trajs, obj_names = autoload_objects(
        pp, f"../data/mogaze/p{dataset}_object_data.hdf5", "../data/mogaze/scene.xml")
    objects = {}
    for i in range(len(obj_names)):
        objects[obj_names[i]] = obj_trajs[i].data

    with open(f'../data/{dataset}/objects.pkl', 'wb') as f:
        pickle.dump(objects, f)

    full_traj = Trajectory()
    full_traj.loadTrajHDF5(f"../data/mogaze/p{dataset}_human_data.hdf5")
    kinematics = HumanKin()
    positions = dict.fromkeys(mykeys)
    orientations = dict.fromkeys(mykeys)

    for key in mykeys:
        positions[key] = []
        orientations[key] = []

    for i in range(full_traj.data.shape[0]):
        kinematics.set_state(full_traj, i)  # set state at frame 100
        for key in mykeys:
            my_id = kinematics.inv_index[key]
            pos = kinematics.get_position(my_id)
            rot = kinematics.get_rotation(my_id)
            rot = np.dot(rot, [0, -1, 0])
            positions[key].append(pos)
            orientations[key].append(rot)

    with open(f'../data/{dataset}/joints_positions.pkl', 'wb') as f:
        pickle.dump(positions, f)

    with open(f'../data/{dataset}/joints_orientations.pkl', 'wb') as f:
        pickle.dump(orientations, f)

    with h5py.File(f"../data/mogaze/p{dataset}_segmentations.hdf5", "r") as segfile:
        with open(f'../data/{dataset}/segmentation.pkl', 'wb') as f:
            pickle.dump(segfile["segments"][0:], f)

    gaze_history = []
    with h5py.File(f"../data/mogaze/p{dataset}_gaze_data.hdf5", "r") as gazefile:
        data = gazefile['gaze'][:]
        calib = gazefile['gaze'].attrs['calibration']

        for i in range(full_traj.data.shape[0]):
            # This segment is copied from humoro/player_pybullet.py
            rotmat = mu.quaternion_matrix(calib)
            rotmat = np.dot(mu.quaternion_matrix(
                objects['goggles'][i][3:7]), rotmat)
            endpos = data[i, 2:5]
            if endpos[2] < 0:
                endpos *= -1  # mirror gaze point if wrong direction
            endpos = np.dot(rotmat, endpos)
            pos = objects['goggles'][i][0:3]
            gazevec = endpos-pos
            gazevec = [x/np.linalg.norm(gazevec) for x in gazevec]
            gaze_history.append(gazevec)

    with open(f'../data/{dataset}/gaze.pkl', 'wb') as f:
        pickle.dump(gaze_history, f)


print("Imported data")
