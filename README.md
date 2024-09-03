# HALLaR: Hybrid Augmented Lagrangian Low-Rank Algorithm

## Overview

The Hybrid Augmented Lagrangian Low-Rank (HALLaR) algorithm [1] by Monteiro, R. D., Sujanani, A. and Cifuentes, D. (2024) is designed to solve large-scale semidefinite programming (SDP) problems efficiently. This repository contains the implementation of the HALLaR algorithm in Python, along with supporting scripts and sample problems to demonstrate its functionality.

## Algorithm Description

The HALLaR algorithm is an inexact augmented Lagrangian method that generates sequences of matrices and Lagrange multipliers through iterative recursions. It leverages a Hybrid Low-Rank (HLR) approach to efficiently solve the augmented Lagrangian subproblems by restricting the solution space to low-rank matrices, significantly reducing the computational complexity. Here, HLR is not yet fully implemented, but instead the minimize functions from scipy.optimize is used. 

## Repository Structure

The repository is organized as follows:

- **main.py**: Implements the HALLaR algorithm.
- **HLR.py**: Contains the shell for the HLR algorithm and supporting functions.
- **MSS_SDP.py**: Defines functions for solving the Maximum Stable Set sample problem.
- **utils.py**: Provides supporting utility functions.
- **create_graphs.py**: A script to create sample graphs for testing.
- **requirements.txt**: Contains the required dependencies. 

### Folders

- **graphs/**: Contains sample graphs used for testing and evaluation.
- **plots/**: Stores plotted convergence results.
- **visualisations/**: Includes additional visualizations of the algorithm's performance.

## Usage

To use the HALLaR algorithm, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/edannas/HALLaR.git
    cd HALLaR
    ```

2. **Install required dependencies**:
    Make sure you have all necessary Python packages installed. You can use `pip` to install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Create sample graphs**:
    Use the `create_graphs.py` script to generate sample graphs:
    ```bash
    python create_graphs.py
    ```

4. **Run the algorithm**:
    Execute `main.py` to run the HALLaR algorithm on the sample problems:
    ```bash
    python main.py
    ```

5. **View results**:
    Plots and visualizations will be saved in the `plots/` and `visualisations/` folders, respectively.

## Contributing

We welcome contributions to enhance the algorithm and its implementation. If you have suggestions or improvements, please feel free to create a pull request. For major changes, please open an issue first to discuss what you would like to change.

## References
Monteiro, R. D., Sujanani, A., & Cifuentes, D. (2024). A low-rank augmented Lagrangian method for large-scale semidefinite programming based on a hybrid convex-nonconvex approach. _arXiv preprint arXiv:2401.12490._
