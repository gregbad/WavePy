"""
Changes made by Greg Badura, © 2019 Georgia Tech Research Corporation, 7/16/2019


This function runs a demo time series simulation in the directory where the files exist. It goes through all three steps
of the wavepy_v2 flow: optimizing the sampling constraints, running the turbulence simulation, and evaluating the
results. The time of the simulation is quite short, but this code will still take on the order of 2 minutes to run. In
addition, windows will pop up that must be closed throughout running this script.
"""

import wavepy_v2 as wp
import sim_evaluation_module
import optimizer_module
import os

# Define the output directory and the location where the simulation report and optimizer report is written
output_dir = "./demo_wavepyv2//"
report_fname = output_dir + "simulation_params.txt"
f_optimizer = output_dir + "optimizer_params.txt"

# Make the output directories if they do not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the desired parameters of the simulation
cn2 = 1e-14
propdist = 3000.0
beam_waist = 	0.05
wavelength = 1e-06
f_curv = -3000.0
log2Nrange_list = [9, 8]

# Run the optimization of sampling parameters for the desired scenario
optimizer_results = optimizer_module.run_optimizer(wavelength, propdist, cn2, beam_waist, f_curv=f_curv,
                                                   beam_type='spherical', c=2, log2N_range=log2Nrange_list)

# Write the optimized sampling parameters to the specified output directory
f_optimizer_out = open(f_optimizer, "w")
for k, v in optimizer_results.items():
    f_optimizer_out.write(str(k) + ' : ' + str(v) + '\n')
f_optimizer_out.close()

# Get the other necessary parameters from the optimizer
side_len = optimizer_results['side_len']
screens = optimizer_results['screens']
Rdx = optimizer_results['Rdx']
dx = optimizer_results['dx']
N = optimizer_results['N']

# Define the time sampling interval of the sensor
sampling_interval_hz = 315
sim_time = 0.02

# Define the wind velocity profiles in meters/second and the wind memory parameter
wind_vx_list = [-1.5, -2, 2.4, 2.2]; wind_vy_list = [2.2, 2, -0.35, 1.5]
alpha_mem = 0.99

# Parameter for the outer scale
L0 = 1000

# Parameter for the spherical diverging Gaussian beam
div_beam_option = 100

# Initialize the wavepy object for the scenario
sim = wp.wavepy(N=N, L0=L0, dx=dx, Rdx=Rdx, simOption=div_beam_option, Cn2=cn2, PropDist=propdist, NumScr=screens,
                W0=beam_waist, f_curv=f_curv, SideLen=side_len, wvl=wavelength)


# Run the phase screen stack evolution
print("Generating a phase screen stack...")
sim.periodicity_corrected_evolve_phase_screen_stack(output_dir,  sim_time, sampling_interval_hz, alpha = alpha_mem,
                                                    power_retained = 0.01, vx_list = wind_vx_list,
                                                    vy_list = wind_vy_list, wind_upper_bound = 20)


# Run turbulence simulations on the phase screen stack just generated
print("Running turbulence simulations on the stack...")
sim.TurbSimEvolvingPhase()

# Evaluate the phase structure function accuracy
sim_evaluation_module.evaluate_phase_structure_function_accuracy_postprocess(report_fname, output_dir)

# Evaluate the power spectral density function vs. theory
sim_evaluation_module.evaluate_PSD_accuracy(report_fname, output_dir)