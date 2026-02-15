import os 
import json
import matplotlib.pyplot as plt
import numpy as np

molecule = 'LiH'
lr=0.001
ds=0.001
datafilenames = [
    'LiH_MinimalistAnsatz_ecp.log',
    ]
for datafilename in datafilenames:

    current_dir = os.getcwd()
    folderpath = current_dir + f'/data_log/'

    ns = 2048  # number of iterations to take into account for the average
    chemical_accuracy = 0.0015936  # [Ha]
    exact_energies = {
                    'Ga': -2.039853,    # source: ferminet-ecp, Li et al.
                    'Kr': -18.472355,   # source: ferminet-ecp, Li et al.
                    'Sc': -46.5536      # source: ferminet-ecp, Li et al.
                    }  # [Ha], lowest cc-pvqz

    exact_energy = exact_energies[molecule]

    ## Unpack the data
    data_log = json.load(open(folderpath + datafilename))
    iters = np.array(data_log['Energy']['iters'])
    tot_energies = np.array(data_log['Energy']['Mean']['real'])
    energy_err = np.array(data_log['Energy']['Sigma'])

    mean_energy = np.mean(tot_energies[-ns:])
    mean_energy_err = np.mean(energy_err[-ns:])
    energy_diff = mean_energy - exact_energy
    print(f"VMC energy = {mean_energy:.6f} ± {mean_energy_err:.4f} Ha")
    print(f'Exact energy: {exact_energy:.6f} Ha')
    print(f'Energy difference with exact: {energy_diff:.6f} Ha or {energy_diff*1000:.6f} mHa')


    plt.plot(iters[:-ns], tot_energies[:-ns], label='VMC', color='#1f77b4', alpha=0.7)
    plt.plot(iters[-ns:], tot_energies[-ns:], label='VMC converged', color='#1f77b4')


plt.axhline(exact_energy + chemical_accuracy, color='k', linestyle='--', zorder=10, label='Chemical accuracy threshold')
plt.axhline(mean_energy, color='r', zorder=10, label='Mean')
plt.axhline(exact_energy, color='k', zorder=10, label='Ferminet-ECP')

plt.ylim(1.001 * exact_energy, 0.995 * exact_energy)

plt.xlabel('Iterations')
plt.ylabel('$E$ [Ha] ')
plt.legend()
outfilename = f'LiH_ecp_optimization_curve.pdf'
output_dir = f'plots/' + outfilename 
plt.savefig(output_dir, dpi=400, bbox_inches='tight')
plt.show()
