import pybullet as p
import pybullet_data
import time

# Start PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the robot
# *** UPDATE THIS PATH to your URDF file ***
robot_id = p.loadURDF("../assets/humanoid.urdf", [0, 0, 1.5])

p.setGravity(0, 0, -9.8)

print("--- INSPECTING ZERO POSE ---")
print("Look at the robot's default T-pose.")
print("This is what your angles must map to.")
print("Press CTRL+C in the terminal to quit.")

try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    p.disconnect()
    print("\nPyBullet disconnected.")