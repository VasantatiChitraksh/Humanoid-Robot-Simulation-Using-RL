# Humanoid Robot Simulation Using RL

This project is an exploration into training a humanoid robot to walk using Reinforcement Learning (RL). The simulation is built using [PyBullet](https://pybullet.org/wordpress/), a physics simulator for robotics, and the ultimate goal is to implement an RL algorithm (like DQN) to enable the robot to learn a stable walking gait.

## ⚠️ Project Status: In Progress

This repository is currently under active development.

* **What Works:** The PyBullet simulation environment is set up, and scripts are available to visualize the humanoid robot's poses.
* **What's Next:** The Reinforcement Learning (DQN) code is present in the repository but **has not yet been integrated or tested**. The core logic for training the robot to walk is the next major milestone.

## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/VasantatiChitraksh/Humanoid-Robot-Simulation-Using-RL](https://github.com/VasantatiChitraksh/Humanoid-Robot-Simulation-Using-RL)
    cd Humanoid-Robot-Simulation-Using-RL
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    (Note: Ensure you have a `requirements.txt` file in your repository)
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run

The current scripts are primarily for visualizing the environment and robot poses.

* **To run the full pose visualization sequence:**
    ```bash
    python main.py
    ```

* **To visualize a single pose or a specific scenario:**
    ```bash
    python main_single.py
    ```

## 🗺️ Roadmap / Future Work

The immediate next step is to transition this from a visualization tool to a full RL training environment.

* [ ] **Integrate DQN:** Connect the existing DQN network code to the PyBullet simulation.
* [ ] **Define State and Action Spaces:** Formally define the observation (state) and action spaces for the RL agent.
* [ ] **Implement Reward Function:** Design and implement a reward function that encourages stable forward walking and penalizes falling.
* [ ] **Train the Model:** Run the training loop and iterate on the reward function and model hyperparameters.
* [ ] **Save and Load Models:** Add functionality to save the trained agent's model weights and load them for inference (i.e., to watch the trained robot walk).
