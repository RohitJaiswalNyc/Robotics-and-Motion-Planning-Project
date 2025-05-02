<h2> Pre-requisites </h2>
Mesh files are large and not version-controlled. You can download them [here](https://sumailsyr-my.sharepoint.com/personal/gkatz01_syr_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fgkatz01%5Fsyr%5Fedu%2FDocuments%2Fmeshes%2Ezip&parent=%2Fpersonal%2Fgkatz01%5Fsyr%5Fedu%2FDocuments&ga=1) (you'll need to be logged into your SU OneDrive account for access). Then, extract the zip archive into a sub-directory of this repository named "meshes".

You also need to install the [PyBullet](https://pybullet.org/wordpress/) simulator.


<h2> How to Run: </h2>
Once PyBullet is installed and the meshes are extracted in the correct location, you can visualize the simulation environment. Run this command from within the top-level folder of the repository:<br\> <br\>

> python simulation.py

# s25rmp: Task and Motion Planning for Robotic Box Placement

## Overview

This project was developed as part of the CIS700: Robotic Motion Planning course. The goal is to use Task and Motion Planning (TAMP) techniques to place colored boxes into their designated target locations using a simulated robotic arm in the PyBullet physics engine.

The robot must account for both position and orientation constraints while executing motion primitives computed via inverse kinematics. Evaluation metrics include accuracy of placement and errors in orientation and position.

## Project Objective

- **Task**: Move boxes from their initial locations to pre-assigned goal regions using a robotic manipulator.
- **Methodology**: Use forward and inverse kinematics, orientation and joint constraints, and simulation-based validation.
- **Tools**: PyBullet for physics simulation and visualization, custom IK/FK solvers in Python.

## Prerequisites

Ensure the following are installed:

- Python 3.7+
- [PyBullet](https://pybullet.org/wordpress/)
- NumPy
- Matplotlib


