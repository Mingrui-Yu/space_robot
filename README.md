# Target Capture of Space Robot

## Introduction

This is a couse project of the Space Robot course at the Department of Automation, Tsinghua University.

In this course project, we propose a scheme for capturing a target satellite out of control using a space robot. The space robot is a service satellite with two robot manipulators. During the on-orbit service, the space robot needs to capture the disabled and spinning target satellite, stop its spinning, restabilize it, and fix it up. We design the robot mechanics and the gripper mechanics, analyze the workspace of the manipulators, and simulate the target capture process based on the PyBullet simulator.


<p align="center">
<iframe width="640" height="360" src="https://player.bilibili.com/player.html?aid=252657952&bvid=BV1jY411p7Nk&cid=471247927&page=1" title="bilibili video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen> </iframe>
</p>


## Dependencies

* Python 3.6
* [PyBullet](https://pybullet.org/wordpress/)



## Usage

First modify the *self.project_dir* in *pybullet/main.py* to the absolute path to this project in your computer.

Run the simulation:
```
cd space_robot

# (activate your python3 env)
python pybullet/main.py
```



* The URDF description files of the space robot, the target satellite, and the gripper are in *urdf/*  folder.
* Some quantitative simulation results are recorded and saved in *results/* folder. To plot these results, run the corresponding scripts in *pybullet/results/*.

