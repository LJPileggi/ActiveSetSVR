# **SVR Active-Set Training Toolbox (MATLAB)**

This repository contains a custom implementation of **Support Vector Regression (SVR)** trained using the **Active-Set method**. The toolbox includes scripts for synthetic data generation, hyperparameter optimisation (model selection), scalability benchmarks, and analysis of initial active-set configurations.

## **Requirements**

* MATLAB (R2020b or later recommended)
* **Optimisation Toolbox**: Required only for comparing results with the built-in quadprog solver in ...QP.m scripts.


# **Main Scripts**

These scripts are designed to be executed directly from the MATLAB command window.

### **1\. ModelSelection.m**

Performs a Grid Search to find the optimal combination of SVR hyperparameters.

* **Usage**: ModelSelection(N, x\_l, x\_u, eps)
* **Parameters**:
  * N: Total number of data points.
  * x\_l, x\_u: Lower and upper bounds for the input range $x$.
  * eps: Gaussian noise level added to the labels.
* **Functionality**: It splits the generated data into Training (80%) and Validation (20%) sets, tests various combinations of $C$, $\\epsilon$, and $\\beta$, and identifies the configuration with the lowest Mean Squared Error (MSE).

### **2\. StartingSetSelection.m / StartingSetSelectionQP.m**

Investigates how the initial choice of Lagrange multipliers ($\\alpha$) affects the convergence of the Active-Set algorithm.

* **Usage**: \[combs, f\_l, time\_l, it\_cost\_l, n\_it\_l\] \= StartingSetSelection(N, x\_l, x\_u, eps, C, e, beta)
* **Parameters**:
  * Includes standard SVR hyperparameters ($C$, $e$, $beta$) to keep the model fixed during testing.
* **Functionality**: Tests different percentages of variables initialized at the lower bound (perc\_l) or upper bound (perc\_u). It returns performance metrics such as total time, iteration count, and function values.

### **3\. Scalability.m / ScalabilityQP.m**

Evaluates the computational performance as the dataset size ($N$) increases.

* **Usage**: \[f\_l, time\_l, it\_cost\_l, n\_it\_l\] \= Scalability(x\_l, x\_u, eps, C, e, beta, perc\_l, perc\_u)
* **Functionality**: Runs the training process on increasing dataset sizes (from 10 to 700 points). It performs 10 attempts for each size to calculate the mean and standard deviation of execution time and cost per iteration.


# **Core Components**

### **SVR.m (Class Definition)**

The central engine of the toolbox. It handles:

* **Kernel Construction**: Implements the RBF (Gaussian) kernel. 
* **Optimisation Problem**: Defines the quadratic objective function and constraints for the SVR dual problem.
* **ActiveSet Method**: A custom solver that iteratively updates the set of active constraints based on KKT conditions and multiplier signs.

### **datagen.m (Utility)**

Generates synthetic data using a combination of sine waves:

$$y \= a\_1\\sin(k\_1x) \+ a\_2\\sin(k\_2x) \+ \\text{noise}$$

This ensures consistent data across all benchmarking scripts.


# **Technical Notes**

* **Initialisation**: The SVR class supports three initialisation modes: "standard" (random), "perc" (percentage-based active sets), and "fix" (specific indices).
* **Precision**: The algorithm uses a tolerance based on $10^{-14} \\times C$ to determine if variables are "touching" the boundaries.
