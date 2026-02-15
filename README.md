# netket-ccECP: Neural Quantum States with Pseudopotentials

---

### 📖 Overview


netket-ccECP is the result of a second semester project (click [here](https://github.com/tjobin/netket-kan) to see the first) in Giuseppe Carleo's Computational Quantum Physics Lab under the supervision of David Linteau. It aimed at integrating a specific type of pseudopotentials, namely ccECP,  into the [NetKet](https://github.com/netket/netket) framework, enabling the use of Variational Monte Carlo simulations for larger systems such as transition metals or, possibly, crystal lattices.

Many thanks must go to ByteDance, the authors of the library [FermiNet_with_ECP](https://github.com/bytedance/FermiNet_with_ECP), for their willingness to leave their work OpenSource, which has been of great help to understand how to implement ccECP in NetKet. This project reused their library as was, to the exception of correcting deprecations. Of course, FermiNet's amazing job, which remains fully available as well, has allowed this project to be fruitful, and for this I am grateful.

#### Disclaimers:
  - David Linteau (@dalin27) wrote the majority of the code and had the kindness to share it with me
  - The actual Neural-Network Quantum States used in this project are not available on this repository, the corresponding publication not being yet published

---

### 🛠 Installation

Prerequisites:

* [NetKet](https://www.netket.org/) - Machine learning framework for quantum many-body problems.
* [JAX](https://github.com/google/jax) - Autograd and XLA for high-performance ML research.
* [Flax](https://github.com/google/flax) - A neural network library for JAX.


```
# Clone the repository
git clone https://github.com/tjobin/netket-kan.git
cd netket-kan

# Install dependencies - made for CPU use !
pip install -r requirements.txt
```

*Note: if you are using GPU acceleration, ensure you have the appropriate version of [NetKet](https://netket.readthedocs.io/en/latest/install.html) and [JAX](https://docs.jax.dev/en/latest/installation.html#installation) installed*

---

### 🚀 Usage

#### Create appropriate folders

```
# Create folder for data and log files:
mkdir data_log

# Create folder for figures
mkdir plots
```
#### Set up the simulation in main.py

- Define the molecule of interest and create the pyscf `mol` object
- Instantiate the ansatz
- Define the sampler hyperparameters and instantiate it
- Define the optimizer hyperparameters and instantiate it
- Instantiate the `netket.operator` object representing the Hamiltonian of the system
- Instantiate the `netket.vqs.MCState` object representing the states sampled by the sampler
- Instantiate the `netket.VMC` object and run the simulation 

### 📊 Results

Chemical accuracy was reached for the transition metals Ga and Kr compared to FermiNet baselines.
  






