from vpython import *

# Create a scene
scene = canvas(title="Cube Simulation", width=600, height=600)

# Create a cube
cube = box(pos=vector(0, 0, 0), size=vector(2, 2, 2), color=color.blue)

# Simulation loop
while True:
    rate(30)  # 30 frames per second
    cube.rotate(angle=0.01, axis=vector(0, 1, 0))  # Rotate around y-axis