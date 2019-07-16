# WavePy V2

Use of this software is provided subject to the included license. It is not intended to be restrictive. Essentially, the creators retain all rights. When using this work or creating derivative works please include the following attribution:

Beck, J., Bekins, C., & Bos, J. P. (2016, May). WavePy: A Python package for wave optics. In SPIE Defense+ Security (pp. 984603-984603). International Society for Optics and Photonics.

The code was ported to python-3 by Greg Badura, © 2019 Georgia Tech Research Corporation.

Several changes were made, including: 
- Correction for propagation plane sampling at the locations of the phase screens
- Ability to evolve wind screens over time to model turbulence propagation through a windy atmosphere
- Ability to use an expanding Gaussian beam, as defined by a radius of curvature
- Evaluation tools for checking the accuracy of the simulation

These changes are outlined in a demo file that is included in this repository.


# Demo of Wavepy v2:

In demo_wavepy_v2.py, I run through a series of operations showing the functionality in wavepy_v2.py. The file demo_wavepy_v2.py uses all of the modules involved in wavepy v2. 

Note that windows will pop up while running this script that must be closed to continue!

In lines 8-12, the major modules involved in wavepy v2 are loaded. The file wavepy_v2.py is the code used to perform turbulence propagation simulations. The sim_evaluation_module.py module is used to evaluate the accuracy of phase screens involved in a simulation. The optimizer_module.py contains functions used to determine the sampling constraints for an accurate simulation. 

In lines 14-16, a local output directory location is defined where all files will be created. The report_fname defines a location where the parameters used to run the simulation will be written after doing a turbulence simulation. The f_optimizer defines a location where the optimization results for the sampling constraints will be written.

In lines 19-20, the output directory is created if it does not already exist.

Lines 23-28 denote the desired Cn^2, laser wavelength, propagation distance, beam radius at the source plane, radius of curvature, and desired N values for the simulation. The parameter log2N_range_list is a list of ordered values, v=log2(N), where N is the side length of the window in pixels. The optimizer will choose the first value in the list for which there is a successful simulation setup found. In this case, because the values are decreasing, we are striving for the highest accuracy simulation possible. In the case of an increasing value list, we would be striving for the fastest simulation possible. 

Lines 31-32 show the function optimizer_module.run_optimizer(...) being run using the input parameters and a spherical beam type. The choice of beam type denotes how the Fried parameter r0 will be calculated. The parameter c denotes how much energy is conserved, with a value c=2 conserving 97% of the total energy. Upon running this simulation, a window of the simulation values and chosen knee values will be displayed. An example of the plot for this simulation setup is shown below. Note that because the values for N=2^8 and N=2^9 were successful in the context of the constraints, the knee values are marked with a star. If they we unsuccessful, they would instead be marked with an 'X'.

In lines 35-38, the dictionary of parameters resulting from this optimization are written out into a text file at the location of f_optimizer

In lines 41-45, several parameters that were determined by the optimizer and that are necessary to run the turbulence simulation are loaded from the dictionary. These include the physical side length of the source field (side_len), the number of screens (screens), the receiver sampling interval (Rdx), the source field sampling interval (dx), and the number of sample intervals per side (N).

Lines 48-49 set up the sampling interval in units of Hz for the receiving sensor, and the total simulation time. These will be important for defined the time interval between phase screen updates. For this short time frame and high sampling interval, a total of 6 timesteps will be genreated

Lines 52 shows the wind velocities at the 4 different phase screens in this simulation in units of m/s. An alternative to this approach is to not define this and to allow the wind velocities to be randomly initialized by the code in the process of generating the phase screen stack.

Line 53 defines the memory scalar parameter for our simulation to be 0.99

Line 56 defines the outer scale of turbulence in units of meters for the simulation

Line 59 defines the beam option to be a diverging Gaussian beam with a radius of curvature that is specified by the user. 

Line 62 creates a wavepy object with the desired input parameters of our simulation setup

Line 68-70 creates a phase screen time stack using the function periodicity_corrected_evolve_phase_screen_stack. Note that this function has two parameters defined that did not require explicit definition in this case. The parameter 'power_retained" was set equal to 0.01, which means that we desire that only 1% of the original phase screen should be retained upon shifting the original phase screen by a full side-length. This parameter should not be made much larger than 0.01 in order to avoid "seams" in the phase screen that are due to the periodic nature of the DFT. The parameter "wind_upper_bound" is only necessary in the case that no wind velocity lists are supplied. It would be used to define an upper bound when randomly initializing the wind velocities at the phase screen locations for all N_screens. 

Upon running the function periodicity_corrected_evolve_phase_screen_stack  phase screens will be created in the output directory as can be seen in the following image. The phase screens are created in sequence and written out as .npy files that can be loaded back into python to run the turbulence simulations over time. They are also written and .png images that can be viewed to make sure that periodicity effects are not affecting phase screen results.

In lines 74-75, the turbulence simulation is run on the phase screen stack. The turbulence fields are written into the same directory as .npy files and .png files. After running the operations, a text file called simulation_params.txt will also be written to the output directory that resembles the following. This function will be necessary for post-processing functions.

Finally, after generating the turbulence simulation, we can evaluate the accuracy.

Line 78 will evaluate the phase structure function accuracy vs. theoretical Kolmogorov and Van-Karman functions. An example plot is shown below for this simulation for the first phase screen location.

Line 81 shows the call to a function for plotting the power spectral density. An example output for the 4 different screens in this simulation are shown below. As can be seen, there is in general good agreement between the theoretical spectrum and the empirical spectrum. Note that this will only be accurate over many averages. The higher frequencies are where the function appears to be least accurate, suggesting that spectral leakage is still affecting our results.
