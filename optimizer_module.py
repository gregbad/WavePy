"""
Created on 5/7/2019

@author: gbadura3-gtri
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy.interpolate import UnivariateSpline
import wavepy_v2 as wp
import json
from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap

def run_optimizer(wvl,
                  PropDist,
                  Cn2,
                  beam_waist,
                  f_curv = -100e3,
                  beam_type = 'spherical',
                  c = 8,
                  log2N_range = [11, 10, 9],
                  max_screens = 15):
    """
    Run an optimization to determine the necessary sampling constraints for the input simulation scenario. The code
    will compare the optimal parameters at the knee of the log2N contour plot for different values of N, as denoted by
    the input list log2N_range. It will return the first N value in the list of log2N_range for which the number of
    screens in the simulation does not exceed max_screens

    :param wvl: the wavelength in meters
    :param PropDist: the propagation distance in meters
    :param Cn2: the cn2 of atmosphere in meters^-2/3s
    :param beam_waist: the beam waist in meters
    :param f_curv: the radius of curvature of the beam. A negative value indicates divergence
    :param beam_type: the type of beam: spherical or planar
    :param c: an energy conservation term. 2 means conserve 97% energy and 4 means conserve 99% energy
    :param log2N_range: the range of values to consider for 2**N. By putting them in descending order, you prioritize
                        a higher sampling rate over quicker simulation
    :param max_screens: the maximum number of screens that you want to use in simulations. Set this lower for quicker
                        but more aliased simulation
    :return: a dictionary containing the optimal parameters and the input conditions
    """

    # Define constants here
    k = 2 * pi / wvl  # optical wavenumber [rad/m]
    R = f_curv  # wavefront radius of curvature [m]


    # Spherical Wave and Plane Wave coherence diameters [m]
    if beam_type == 'spherical':
        r0 = (0.423 * k ** 2 * Cn2 * 3 / 8 * PropDist) ** (-3 / 5)
    else:
        r0 = (0.423 * k ** 2 * Cn2 * PropDist) ** (-3 / 5)


    # Calculate the tranverse coherence length for the purpose of calculating expected beam width
    rho_0 = r0/2.1

    # User toggle for plane or spherical wavefront
    coherence_diam = r0

    # Calculate the diameter of the receiver plane of interest using the equations for the long term beam wandering
    D_beam = np.sqrt(2) * beam_waist # the aperture diameter, per page 196 of the EOIR textbook
    D1 =  D_beam
    BW_diffraction = (4 * PropDist ** 2.0) / ((k * D_beam) ** 2.0)
    BW_focusing = ((D_beam / 2) ** 2.0) * ((1 - PropDist / R) ** 2.0)
    BW_turbulence_spread = (4 * PropDist ** 2.0) / ((k * rho_0) ** 2.0)

    print("Input beam diameter: ", D_beam)
    print("Expected beam width due to diffraction and focusing: ",
          np.sqrt(2 * (BW_diffraction + BW_focusing )))
    print("Expected beam width due to diffraction, turbulence and focusing: ",
          np.sqrt(2 * (BW_diffraction + BW_focusing + BW_turbulence_spread)))
    print("Spread due to turbulence: ")
    print(BW_turbulence_spread)
    print()

    # Diameter for the observation aperture for long term averaging, from equation (2.154) of EOIR book
    # DRx = np.sqrt(2 * (BW_diffraction + BW_focusing + BW_turbulence_spread))

    # Changing DRx to the vacuum width
    # DRx =  np.sqrt(2 * (BW_diffraction + BW_focusing)) # by definition, the beam radius actually
    # DRx = 2* np.sqrt((BW_diffraction + BW_focusing)) # by definition, the beam diameter
    DRx = 2*np.sqrt((BW_diffraction + BW_focusing + BW_turbulence_spread))
    # DRx = np.sqrt(2.4 * (BW_diffraction + BW_focusing))  # a trade off for computational purposes


    # log-amplitude variance
    p = np.linspace(0, int(PropDist), 1000)
    rytov = 0.563 * k ** (7 / 6) * sum(Cn2 * (1 - p / PropDist) ** (5 / 6) * p ** (5 / 6) * (p[1] - p[0]))

    # screen properties
    # Define the r0's of each phase screen using a standard minimization algorithm.
    # There are potential divide by zero warnings that are then addressed by the code.
    NumScr = 11  # number of screens
    A = np.zeros((2, NumScr))  # matrix
    alpha = np.arange(0, NumScr) / (NumScr - 1)
    A[0, :] = alpha ** (5 / 3)
    A[1, :] = (1 - alpha) ** (5 / 6) * alpha ** (5 / 6)
    b = np.array([coherence_diam ** (-5 / 3), rytov / 1.33 * (k / PropDist) ** (5 / 6)])
    # initial guess
    x0 = (NumScr / 3 * coherence_diam * np.ones((NumScr, 1))) ** (-5 / 3)
    # objective function, @ is matrix multiply
    fun = lambda X: np.sum((A @ X.flatten() - b) ** 2)
    # constraints
    x1 = np.zeros((NumScr, 1))
    rmax = 0.1  # maximum Rytov number per partial prop according to Martin/Flatte
    x2 = rmax / 1.33 * (k / PropDist) ** (5 / 6) / A[1, :]
    x2[A[1, :] == 0] = 50 ** (-5 / 3)  # address divide by zero

    res = minimize(fun, x0, bounds=Bounds(x1, x2))
    soln = res.x
    # check screen r0s
    r0scrn = soln ** (-3 / 5)
    r0scrn[np.isinf(r0scrn)] = 1e6  # address divide by zero
    # check resulting coherence_diam & rytov with minimization solution
    # too few phase screens will cause these number to disagree
    bp = A @ soln.flatten()
    compare1 = [bp[0] ** (-3 / 5), bp[1] * 1.33 * (PropDist / k) ** (5 / 6)]
    compare2 = [coherence_diam, rytov]
    # print(compare1, compare2)


    # Account for conservation of energy term in the source plane, the receiver plane has already been accounted for
    D1p = D1 + c * wvl * PropDist / coherence_diam

    # Changing the accounting for turbulence spreed here
    # DRxp = DRx
    DRxp = DRx + c * wvl * PropDist / coherence_diam

    print("Expected beam width due to the conversation of energy after turbulent prop: ")
    print(DRxp)
    print()

    # Now perform the minimization on constraints 1-4
    delta1 = np.linspace(1.1 * wvl * PropDist / DRxp/1000, 1.1 * wvl * PropDist / DRxp, 1000)
    deltan = np.linspace(1.1 * wvl * PropDist / D1p/1000, 1.1 * wvl * PropDist / D1p, 1000)

    # constraint 1
    deltan_max = -DRxp / D1p * delta1 + wvl * PropDist / D1p
    # constraint 3
    Rdxmin3 = (1 + PropDist / R) * delta1 - wvl * PropDist / D1p
    Rdxmax3 = (1 + PropDist / R) * delta1 + wvl * PropDist / D1p


    # Derive the knee curve for each log2N value
    delta1_knee_list = []
    deltan_knee_list = []
    amp_value = 1.0
    for log2N in log2N_range:
        N_curr = 2**log2N
        deltan_constraint = (amp_value*wvl*PropDist/(2*delta1) + amp_value*DRxp/2)/(N_curr - amp_value*D1p/(2*delta1))

        # If deltan_constraint has values less than zero, then ignore those values
        valid_values = deltan_constraint > 0
        delta1_temp = delta1[valid_values]
        deltan_constraint = deltan_constraint[valid_values]

        # Find the default knee value index
        min_deltan = np.nanmin(deltan_constraint)
        max_deltan = np.nanmax(deltan_constraint)
        min_delta1 = np.min(delta1_temp)
        max_delta1 = np.max(delta1_temp)
        default_knee_logval_idx = np.nanargmin((deltan_constraint/max_deltan - min_deltan/max_deltan)**2.0 +
                                       (delta1_temp/max_delta1 - min_delta1/max_delta1)** 2.0)

        # Iterate through the other possible knee values and find if any have a kee that will maximize layer thickness
        knee_logval_idx = default_knee_logval_idx
        max_sampling_dxdn = np.min([delta1_temp[default_knee_logval_idx],
                                    deltan_constraint[default_knee_logval_idx]])
        for idx in range(0, np.size(deltan_constraint)):
            if np.min([delta1_temp[idx], deltan_constraint[idx]]) > max_sampling_dxdn:
                max_sampling_dxdn = np.min([delta1_temp[idx], deltan_constraint[idx]])
                knee_logval_idx = idx


        delta1_knee_list.append(delta1_temp[knee_logval_idx])
        deltan_knee_list.append(deltan_constraint[knee_logval_idx])

    # Debug: print the knee lists for inspection
    # print("Knee lists for delta1 and deltan: ")
    # print(delta1_knee_list)
    # print(deltan_knee_list)


    # Now iterate through the knee values and calculate the constraints on 1, 3 and the screens and make sure that they are valid
    c1_list = []
    c3_list = []
    c_screen_list = []
    num_screen_list = []
    max_layer_thickness_list = []
    for idx in range(len(log2N_range)):
        d1_idx = delta1_knee_list[idx]
        dn_idx = deltan_knee_list[idx]

        # Constraint 1:
        c1_deltan = (-DRxp / D1p * d1_idx + wvl * PropDist / D1p) >= dn_idx
        c1_list.append(c1_deltan)

        # Constraint 3:
        c3_deltan =  ((1 + PropDist / R) * d1_idx - wvl * PropDist / D1p) <= dn_idx and \
                     ((1 + PropDist / R) * d1_idx + wvl * PropDist / D1p) >= dn_idx
        c3_list.append(c3_deltan)

        # Final constraint: Is the minimum number of screens less than the desired max number
        N = 2**log2N_range[idx]
        zmax = (min(d1_idx, dn_idx) ** 2) * N / wvl  # mathematical placeholder
        max_layer_thickness_list.append(zmax)

        numScrMin = np.ceil(PropDist / zmax) + 2 # 1 , incremented beyond the minimum
        num_screen_list.append(numScrMin)

        c_screen = numScrMin <= max_screens
        c_screen_list.append(c_screen)

    # Debug:
    print("Min number of screens for log2N list", str(log2N_range) , " : ")
    print(num_screen_list)
    print()


    # Using the descending order of our log2N_range list, return the maximum value that satisfies all constraints
    # while also satisfying the minimum number of screens
    constraint_list = np.logical_and(np.logical_and(c1_list, c3_list), c_screen_list)
    where_constraints_satisfied = np.where(constraint_list == True)[0]
    optimal_constraint_dict = {}
    optimal_constraint_dict["cn2"] = Cn2
    optimal_constraint_dict["propdist"] = PropDist
    optimal_constraint_dict["beam_waist"] = beam_waist
    optimal_constraint_dict["wavelength"] = wvl
    optimal_constraint_dict["f_curv"] = R

    if np.size(where_constraints_satisfied) == 0:
        print("There is no value for which the maximum number of screens and constraints 1-4 are satisfied for this"
              "scenario. Please look at the plot and revise your simulation setup. ")
        optimal_constraint_dict["success"] = False
    else:
        optimal_constraint_dict["success"] = True
        first_constraint_idx = where_constraints_satisfied[0]
        print("A satisfactory simulation scenario was found. Values of first occurence are: ")
        print("N = ", 2**log2N_range[first_constraint_idx])
        optimal_constraint_dict["N"] = 2**log2N_range[first_constraint_idx]
        print("dx = ", delta1_knee_list[first_constraint_idx])
        optimal_constraint_dict["dx"] = delta1_knee_list[first_constraint_idx]
        print("Rdx = ", deltan_knee_list[first_constraint_idx])
        optimal_constraint_dict["Rdx"] = deltan_knee_list[first_constraint_idx]
        print("Min # Screens = ", num_screen_list[first_constraint_idx])
        optimal_constraint_dict["screens"] = int(num_screen_list[first_constraint_idx])


        # print("Side Len = ", np.max([delta1_knee_list[first_constraint_idx] * 2**log2N_range[first_constraint_idx],
        #                              deltan_knee_list[first_constraint_idx] * 2**log2N_range[first_constraint_idx]]))
        # optimal_constraint_dict["side_len"] = np.max([delta1_knee_list[first_constraint_idx] * 2**log2N_range[first_constraint_idx],
        #                                         deltan_knee_list[first_constraint_idx] * 2**log2N_range[first_constraint_idx]])

        # Set sidelen to be the initial distance
        print("Side Len = ", delta1_knee_list[first_constraint_idx] * 2**log2N_range[first_constraint_idx])
        optimal_constraint_dict["side_len"] = delta1_knee_list[first_constraint_idx] * 2**log2N_range[first_constraint_idx]

        print()


    # Now plot the contours along with the valid constraints for our case
    plt.figure(1)
    plt.plot(delta1, deltan_max, 'b+')
    plt.plot(delta1, Rdxmin3, 'r-')
    plt.plot(delta1, Rdxmax3, 'r+')
    X, Y = np.meshgrid(delta1, deltan)
    log2N_range.sort()
    N2_contour = (wvl * PropDist + D1p * Y + DRxp * X) / (2 * X * Y)
    contours = plt.contour(delta1, deltan, np.log2(N2_contour), log2N_range)
    plt.clabel(contours, inline=True)

    # Plot the knee points with a marker to denote if it satisfies our constraints
    for idx in range(len(log2N_range)):

        # If it satisfies all three constraints, give it a star marker. Otherwise, give it a x marker
        if c1_list[idx] and c3_list[idx] and c_screen_list[idx]:
            marker_constraint = "*"
        else:
            marker_constraint = "x"

        plt.scatter(delta1_knee_list[idx], deltan_knee_list[idx], marker=marker_constraint, s=40)

    # plt.colorbar()
    plt.axis([0, delta1[-1], 0, deltan[-1]])
    plt.title('Constraints 1, 2, 3 for point source problem')
    plt.xlabel('dx [m]')
    plt.ylabel('dn [m]')
    # plt.savefig('screen-params.png')

    plt.show()

    return optimal_constraint_dict

def run_beam_propagation_simulation(optimizer_results,
                                    output_dir = "",
                                    cmap_txt_file = None):
    """
    Accept the dictionary of optimal parameters from the optimizer function. Output the imagery to .png files if
    necessary at the end.

    :param optimizer_results: a dictionary containing optimization results
    :param output_dir: the output directory to write png files to
    :param cmap_txt_file: a text file containing the colormap that we want to use
    :return: no return as of now
    """


    # option made by Greg to signify that we want to use a focused gaussian beam
    sim_focus = 100

    # Get the colormap input by the user. If none, use viridis
    if cmap_txt_file is None:
        use_cmap = cm.viridis
    else:
        cm_data = np.loadtxt(cmap_txt_file)
        use_cmap = LinearSegmentedColormap.from_list('custom cmap', cm_data)


    # Get the extents of the sampling array
    N =  optimizer_results["N"]
    Rdx = optimizer_results["Rdx"]
    sampling_extent = [-(N*Rdx)/2., +(N*Rdx)/2., -(N*Rdx)/2., +(N*Rdx)/2.,]


    # Generate the simulation object
    sim = wp.wavepy(N=optimizer_results["N"],
                    L0=1e4,
                    dx=optimizer_results["dx"],
                    Rdx=optimizer_results["Rdx"],
                    simOption=sim_focus,
                    Cn2=optimizer_results["cn2"],
                    PropDist=optimizer_results["propdist"],
                    NumScr=optimizer_results["screens"],
                    W0=optimizer_results["beam_waist"],
                    f_curv=optimizer_results["f_curv"],
                    SideLen=optimizer_results["side_len"],
                    wvl=optimizer_results["wavelength"])

    # Perform a propagation through turbulence and visualize the results
    sim.TurbSim_layers()
    fig_turb, ax_turb = plt.subplots(1,1)
    I_turb = np.abs(sim.Output) ** 2.0
    ax_turb.imshow(I_turb, cmap = use_cmap, extent = sampling_extent)
    fig_turb.savefig(output_dir + "turb.png")

    # Do a vaccuum propagation on focused vs collimated beam
    I_vaccuum = np.abs(sim.VacuumProp()) ** 2.0
    fig_vac, ax_vac = plt.subplots(1, 1)
    ax_vac.imshow(I_vaccuum, cmap = use_cmap, extent = sampling_extent)
    fig_vac.savefig(output_dir + "vacuum.png")

    # Save the raw data in the form of a numpy array
    np.save(output_dir + "turb_sim.npy", I_turb)

    # Debug: Show the plots
    # plt.show()

    # Return the intensity turbulence for averaging
    return I_turb



if __name__ == '__main__':

    # Specify the input parameters of the simulation
    Cn2 = 1e-14
    wvl = 1e-6  # optical wavelength [m]
    PropDist = 3e3  # propagation distance [m]
    W0 = 5e-2 # beam radius at the 1/e points
    fcurv = -PropDist
    log2Nrange_list = [9,8]

    # Get the dictionary of optimal results
    optimizer_results = run_optimizer(wvl,
                                      PropDist,
                                      Cn2,
                                      W0,
                                      f_curv=fcurv,
                                      beam_type='spherical',
                                      c=2,
                                      log2N_range=log2Nrange_list)

    # Write the output directory to a specified output directory
    output_dir = "H:\projects\wavepy\wavepy_v2_runs\collected_energy_sim\cn2_1e-14\\"
    fout = output_dir + "opt_params.txt"
    fo = open(fout, "w")
    for k, v in optimizer_results.items():
        fo.write(str(k) + ' : ' + str(v) + '\n')
    fo.close()