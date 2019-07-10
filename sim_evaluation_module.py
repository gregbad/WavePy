import numpy as np
import yaml
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import wavepy_v2 as wp

def MakePupil(D_eval, side_len, N):
    """
    Create a pupil at the receiver plane to evaluate the structure function that accounts for receiver sampling effects

    :param D_eval: the diameter of the pupil in meters
    :param side_len: the sidelength of the receiver plane in meters
    :param N: the number of discrete intervals at the receiver
    :return: a pupil function that can be used to mask the receiver plane to the desirec aperture
    """

    # Target pupil creation
    boundary1 = -(side_len / 2)  # sets negative coord of sidelength
    boundary2 = side_len / 2  # sets positive coord of sidelength

    A = np.linspace(boundary1, boundary2, N)  # creates a series of numbers evenly spaced between
    # positive and negative boundary
    A = np.array([A] * N)  # horizontal distance map created

    base = np.linspace(boundary1, boundary2, N)  # creates another linspace of numbers
    set_ones = np.ones(N)  # builds array of length N filled with ones
    B = np.array([set_ones] * N)
    for i in range(0, len(base)):
        B[i] = B[i] * base[i]  # vertical distance map created

    A = A.reshape(N, N)
    B = B.reshape(N, N)  # arrays reshaped into matrices

    x_coord = A ** 2
    y_coord = B ** 2

    rad_dist = np.sqrt(x_coord + y_coord)  # now radial distance has been defined

    mask = []
    for row in rad_dist:
        for val in row:
            if val < D_eval:
                mask.append(1.0)
            elif val > D_eval:
                mask.append(0.0)
            elif val == D_eval:
                mask.append(0.5)
    mask = np.array([mask])
    mask = mask.reshape(N, N)  # mask created and reshaped into a matrix

    return mask  # returns the pupil mask as the whole function's output


def structure_function_over_time(report_filename,
                                 sim_result_directory,
                                 D_receiver_pupil = None):
    """
    Evaluate the accuracy of the turbulence simulation by computing the structure function at the receiver plane using
    a mutual coherence function approach. Note: This will only be accurate over many different simulations and some
    disagreement should be expected over a single turbulence simulation

    :param report_filename: the filename including path for the output report of the turbulence simulation
    :param sim_result_directory: the directory in which the turbulence output simulation files over the timesteps have
                                 been written out
    :param D_receiver_pupil: the receiver pupil diametre in meters for cropping purposes
    :return:
    """

    with open(report_filename, 'r') as f:
        sim_dict = yaml.load(f)

    # Get the necessary constants to define the simulation
    tsteps = sim_dict["Timesteps"]
    Rdx = sim_dict["Receiver Rdx"]
    N = sim_dict["N"]
    r0 = sim_dict["r0"]
    L0 = sim_dict['L0']
    l0 = sim_dict['l0']

    # Constants for performing calculations of the MCF
    delta_f = 1/(N*Rdx)

    # Get the sidelength at the receiver and create a mask of the same size as the receiver pupil
    side_len = N*Rdx
    if D_receiver_pupil is None:
        D_receiver_pupil =  r0*5.
    aperture_mask = MakePupil(D_receiver_pupil,
                              side_len,
                              N)

    # Figure for plotting the structure function over time
    fig, ax = plt.subplots(figsize=(9, 6))
    color = cm.jet(np.linspace(0, 1, tsteps))

    # Use a radial average out to the cutoff in order to get the empirical D
    mesh_spacing = np.arange(0, N)
    X, Y = np.meshgrid(mesh_spacing, mesh_spacing)
    r = np.hypot(X - N / 2, Y - N / 2)
    r[int(N / 2), int(N / 2)] = 0
    rbin = (N * r / r.max()).astype(np.int)
    bin_sampling = (r.max()) * Rdx * np.arange(1, rbin.max() + 1) / rbin.max()

    # For each individual timestep, evaluate the structure function for comparison with the theoretical sim
    bin_sampling_cutoff = 0
    where_within_ap = 0
    mcf_sum = 0
    D_sim_array = []
    for t in range(0, tsteps):
        fname_turbsim_t = sim_result_directory + "_turbsim_t" + "{:04d}".format(t) + ".npy"
        turbsim_t = np.load(fname_turbsim_t)

        # Calculate the masked receiver plane autocorrelation function
        U_r = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(turbsim_t * aperture_mask))) * (Rdx)**2.0
        U_r_autocorr = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(np.conj(U_r) * U_r))) * (delta_f * N) ** 2.0

        # Compensate for the effects of the masking at the receiver aperture
        maskcorr = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(
            abs(np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(aperture_mask)))) * (Rdx) ** 2.0))) * (N * delta_f) ** 2.0

        plt.imshow(np.abs(U_r_autocorr))
        plt.show()
        plt.imshow(np.abs(maskcorr))
        plt.show()


        # idx = maskcorr != 0
        c = U_r_autocorr / maskcorr * aperture_mask

        # Note that c represents the mutual coherence function. Use equation 9.31 and 9.32 from Schmitt to compute
        # the structure function for the sampling grid
        D_r_t = -2 * np.log(np.abs(c) / np.abs(c[int(N / 2), int(N / 2)]))

        # Sum the MCF function
        mcf_sum = mcf_sum + c

        # Bin the data to get the radial mean values of the 2d structure function as a 1d array
        D_1d = scipy.ndimage.mean(D_r_t, labels=rbin, index=np.arange(1, rbin.max() + 1))

        # Calculate where it is less than the aperture and plot
        where_within_ap = bin_sampling <= D_receiver_pupil
        bin_sampling_cutoff = bin_sampling[where_within_ap]
        D_1d_cutoff = D_1d[where_within_ap]

        # Append to holder array
        D_sim_array.append(D_1d_cutoff)

        # Plot it according to color
        # color_t = color[t]
        # ax.plot(bin_sampling_cutoff , D_1d_cutoff, c=color_t)

    # Calculate the averaged MCF factor
    MCF_sim = np.abs(mcf_sum) / np.abs(mcf_sum[int(N/2), int(N/2)])
    D_sim_avg = -2 * np.log(np.abs(MCF_sim) / np.abs(MCF_sim[int(N / 2), int(N / 2)]))

    # plt.imshow(np.abs(MCF_sim))
    # plt.show()
    # plt.imshow(np.abs(D_sim_avg))
    # plt.show()

    D_sim_1d = scipy.ndimage.mean(D_sim_avg, labels=rbin, index=np.arange(1, rbin.max() + 1))
    D_sim_1d_cutoff = D_sim_1d[where_within_ap]
    ax.plot(bin_sampling_cutoff/r0, D_sim_1d_cutoff, 'red', label = "Sim Average MCF")

    # Compute the max and min bounds on the array of stucture functions over time
    D_sim_array = np.asarray(D_sim_array)
    average_D = np.average(D_sim_array, axis=0)
    max_D = np.max(D_sim_array, axis=0)
    min_D = np.min(D_sim_array, axis=0)

    ax.plot(bin_sampling_cutoff/r0, min_D, 'g--', label = 'Sample Lower Bound')
    ax.plot(bin_sampling_cutoff/r0, max_D, 'k--', label = 'Sample Upper Bound')
    ax.plot(bin_sampling_cutoff/r0, average_D, 'b', label = 'Sample MCF Average')

    # Compute the theoretical D
    D_kolmog = 6.88 * (bin_sampling_cutoff/r0) ** (5. / 3.)
    # ax.plot(bin_sampling_cutoff/r0, D_kolmog, 'r', label = 'Kolm (theoretical')

    k0 = 2 * np.pi / L0
    D_mVK = 7.75 * ((r0) ** (-5. / 3.)) * (l0 ** (-1. / 3.)) * (bin_sampling_cutoff ** 2.) * \
            ((1 / (1 + 2.03 * (bin_sampling_cutoff ** 2.) / (l0 ** 2.0)) ** (1. / 6.)) - 0.72 * ((k0 * l0) ** (1. / 3.)))
    ax.plot(bin_sampling_cutoff/r0, D_mVK, 'orange', label = 'MvK (theoretical)')

    ax.legend()
    ax.set_xlabel("$\Delta r$/$r_0$")
    ax.set_ylabel("$D_\Phi$(|$\Delta r$|) [rad$^2$]")

    plt.show()


def evaluate_phase_structure_function_accuracy(input_N,
                                             input_dx,
                                             input_cn2,
                                             num_screen_draws = 20,
                                             input_propdist = 3e3,
                                             input_wave = 1e-6,
                                             input_num_subharmonics = 5,
                                             input_L0 = 1e3,
                                             input_n_screen_sim = 4):
    """
    Generate many different phase screens for a simulation setup and compare the statistical structure funciton of the
    phase screen to theoretical fits for a

    :param input_N: N for the simulation
    :param input_dx: sampling at the source plane for the simulation
    :param input_cn2: the turbulence of the simulation
    :param num_screen_draws: the number of random screens draws for computing statistics
    :param input_propdist: the propagation distance of the simulation
    :param input_wave: the wavelength in meters for sim
    :param input_num_subharmonics: the number of subharmonics for low frequency screen draws
    :param input_L0: the outer scale of the simulation
    :param input_n_screen_sim: the number of screens used in the simulation for propagating a beam
    :return:
    """


    # Create a wavepy instance specified by the input parameters for creating a phase screen instance
    sim = wp.wavepy(N=input_N,
                    L0=input_L0,
                    dx=input_dx,
                    Rdx=input_dx,
                    Cn2=input_cn2,
                    PropDist=input_propdist,
                    NumScr=input_n_screen_sim,
                    W0=5, # Parameter doesn't matter for this context
                    SideLen=input_N * input_dx,
                    wvl=input_wave)

    # Get the r0 for a single screen in the simulation setup, which is defined by the number of screens used
    r0_scrn_sim = sim.r0scrn

    # # Constants for performing calculations of the MCF
    # delta_f = 1 / (input_N * input_dx)

    # Sum the structure functions
    sum_Dr = 0

    # Create the desired number of phase screen draws and calculate the autocorrelation function
    for nscr in range(0, num_screen_draws):
        phz_hi = sim.PhaseScreen(input_dx)
        phz_lo = sim.SubHarmonicComp(input_num_subharmonics, input_dx)

        fig, ax = plt.subplots(1,3)
        ax[0].imshow(phz_hi)
        ax[1].imshow(phz_lo)
        ax[2].imshow(phz_lo + phz_hi)
        plt.show()

        phz =  phz_lo + phz_hi

        # # Now calculate the autocorrelation of the phase screen
        # # Calculate the masked receiver plane autocorrelation function
        # F_phz = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(phz))) * (input_dx) ** 2.0
        # B_r = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(np.conj(F_phz) * F_phz))) * (delta_f * input_N) ** 2.0
        # sum_B_r = sum_B_r + B_r
        #
        # # Calculate the expected structure function by using the relationship between D and B
        # D_r = 2*(np.abs(B_r[int(input_N/2), int(input_N/2)]) - np.abs(B_r))

        # Calculate the structure function using the wavepy function
        D_bos = sim.StructFunc(phz)


        sum_Dr = sum_Dr + D_bos


    # Compute the average structure function
    D_r_avg = sum_Dr / num_screen_draws


    # print(input_N / 4/2)
    #
    # plt.imshow(D_r_avg)
    # plt.show()

    # Calculate the radial average
    x = np.arange(-int(input_N/2), int(input_N/2))
    X, Y = np.meshgrid(x, x)
    R = np.sqrt((X*input_dx)**2 + (Y*input_dx)**2)
    R[int(input_N / 2), int(input_N / 2)] = 0
    rbin = (input_N * R / R.max()).astype(np.int)
    bin_sampling = (R.max()) *  np.arange(0, rbin.max() + 1) / rbin.max()
    cutoff_aperture = (input_N * input_dx / 4)/2
    where_within_aperture = bin_sampling < cutoff_aperture
    bin_sampling_aperture = bin_sampling[bin_sampling < cutoff_aperture]
    D_avg_1d = scipy.ndimage.mean(D_r_avg, labels=rbin, index=np.arange(0, rbin.max() + 1))
    D_avg_1d_cutoff = D_avg_1d[where_within_aperture]


    # Compute the theoretical structure function
    k0 = 2 * np.pi / input_L0
    l0 = 1e-3
    D_mVK_2d = 7.75 * ((r0_scrn_sim) ** (-5. / 3.)) * (l0 ** (-1. / 3.)) * (R ** 2.) * \
            ((1 / ((1 + 2.03 * (R ** 2.) / (l0 ** 2.0)) ** (1. / 6.))) - 0.72 * ((k0 * l0) ** (1. / 3.)))
    D_Kolm_2d = 6.88*(R/r0_scrn_sim)**(5/3)

    # Calculate the 1D form of the structure function out to the cutoff aperture point
    D_mVK_1d = 7.75 * ((r0_scrn_sim) ** (-5. / 3.)) * (l0 ** (-1. / 3.)) * (bin_sampling_aperture ** 2.) * \
            ((1 / ((1 + 2.03 * (bin_sampling_aperture ** 2.) / (l0 ** 2.0)) ** (1. / 6.))) - 0.72 * ((k0 * l0) ** (1. / 3.)))
    D_Kolm_1d = 6.88 * (bin_sampling_aperture / r0_scrn_sim) ** (5 / 3)

    plt.plot(bin_sampling_aperture, D_avg_1d_cutoff, 'r', label= 'Simulation')
    plt.plot(bin_sampling_aperture, D_mVK_1d, 'k--', label = 'MvK')
    plt.plot(bin_sampling_aperture, D_Kolm_1d, 'b--', label = 'Kolmogorov')
    plt.legend()
    plt.xlabel("$\Delta r$ [meters]")
    plt.ylabel("$D_\Phi$(|$\Delta r$|) [rad$^2$]")
    plt.show()


def evaluate_phase_structure_function_accuracy_postprocess(report_filename,
                                                           sim_result_directory):
    """
    Evaluate the phase structure function accuracy on a series of phase screens that were generated for the purpose of
    running a turbulence time series simulation

    Note: I have never had much confidence in this metric, as the scaling of the phase screens greatly affects the
          accuracy of the results

    :param report_filename: the filename including path for the output report of the turbulence simulation
    :param sim_result_directory: the directory in which the turbulence output simulation files over the timesteps have
                                 been written out

    :return: no return at the moment, just display plots
    """

    with open(report_filename, 'r') as f:
        sim_dict = yaml.load(f)

    # Get the necessary constants to define the simulation
    tsteps = sim_dict["Timesteps"]
    dx = sim_dict["Source dx"]
    N = sim_dict["N"]
    r0 = sim_dict["r0"]
    L0 = sim_dict['L0']
    l0 = sim_dict['l0']
    cn2 = float(sim_dict['Cn2'])
    wave = float(sim_dict['wavelength'])
    propdist = sim_dict['Propagation Dist']
    nscreens = sim_dict['Num Screens']
    Rdx = sim_dict["Receiver Rdx"]


    # Create a wavepy instance specified by the input parameters for creating a phase screen instance
    sim = wp.wavepy(N=N,
                    L0=L0,
                    dx=dx,
                    Rdx=Rdx,
                    Cn2=cn2,
                    PropDist=propdist,
                    NumScr=nscreens,
                    W0=5, # Parameter doesn't matter for this context
                    SideLen = N * dx,
                    wvl = wave)

    # Get the r0 for a single screen in the simulation setup, which is defined by the number of screens used
    r0_scrn_sim = sim.r0scrn


    # Constants for performing calculations of the MCF
    delta_f = 1 / (N * dx)


    # Calculate the alpha parameter between sampling planes
    dzProps = np.ones(nscreens + 2) * (propdist / nscreens)
    dzProps[0:2] = 0.5 * (propdist / nscreens)
    dzProps[nscreens:nscreens + 2] = 0.5 * (propdist / nscreens)
    PropLocs = np.zeros(nscreens + 3)
    for zval in range(0, nscreens + 2):
        PropLocs[zval + 1] = PropLocs[zval] + dzProps[zval]
    ScrnLoc = np.concatenate((PropLocs[1:nscreens],np.array([PropLocs[nscreens + 1]])), axis=0)
    FracPropDist = PropLocs / propdist
    dx_phase_screens = (Rdx - dx) * FracPropDist + dx

    # Cutoff aperture radius for the structure function calculation
    cutoff_aperture = (N * dx ) / 4

    # Create the desired number of phase screen draws and calculate the autocorrelation function
    for nscr in range(0, nscreens):
        # Calculate the sampling at the current plane
        dx_scr = dx_phase_screens[nscr]

        # Sum the structure functions over all time steps
        sum_Dr = 0

        # Sum the structure functions of the current phase screen
        for t in range(0, tsteps):
            fname_scr_draw = sim_result_directory + "_scr" + str(nscr) + "_t" + "{:04d}".format(t) + ".npy"
            scr_draw = np.load(fname_scr_draw)

            D_bos = sim.StructFunc(scr_draw)
            sum_Dr = sum_Dr + D_bos

        # Compute the average structure function
        D_r_avg = sum_Dr / tsteps


        # Calculate the radial average
        x = np.arange(-int(N / 2), int(N / 2))
        X, Y = np.meshgrid(x, x)
        R = np.sqrt((X * dx_scr) ** 2 + (Y * dx_scr) ** 2)
        R[int(N / 2), int(N / 2)] = 0
        rbin = (N * R / R.max()).astype(np.int)
        bin_sampling = (R.max()) * np.arange(0, rbin.max() + 1) / rbin.max()
        where_within_aperture = bin_sampling < cutoff_aperture
        bin_sampling_aperture = bin_sampling[bin_sampling < cutoff_aperture]
        D_avg_1d = scipy.ndimage.maximum(D_r_avg, labels=rbin, index=np.arange(0, rbin.max() + 1))
        D_avg_1d_cutoff = D_avg_1d[where_within_aperture]


        # Compute the theoretical structure function
        k0 = 2 * np.pi / L0
        l0 = 1e-3
        D_mVK_2d = 7.75 * ((r0_scrn_sim) ** (-5. / 3.)) * (l0 ** (-1. / 3.)) * (R ** 2.) * \
                ((1 / ((1 + 2.03 * (R ** 2.) / (l0 ** 2.0)) ** (1. / 6.))) - 0.72 * ((k0 * l0) ** (1. / 3.)))
        D_Kolm_2d = 6.88*(R/r0_scrn_sim)**(5/3)

        # Calculate the 1D form of the structure function out to the cutoff aperture point
        D_mVK_1d = 7.75 * ((r0_scrn_sim) ** (-5. / 3.)) * (l0 ** (-1. / 3.)) * (bin_sampling_aperture ** 2.) * \
                ((1 / ((1 + 2.03 * (bin_sampling_aperture ** 2.) / (l0 ** 2.0)) ** (1. / 6.))) - 0.72 * ((k0 * l0) ** (1. / 3.)))
        D_Kolm_1d = 6.88 * (bin_sampling_aperture / r0_scrn_sim) ** (5 / 3)

        plt.plot(bin_sampling_aperture, D_avg_1d_cutoff, 'r', label= 'Simulation')
        plt.plot(bin_sampling_aperture, D_mVK_1d, 'k--', label = 'MvK')
        plt.plot(bin_sampling_aperture, D_Kolm_1d, 'b--', label = 'Kolmogorov')
        plt.legend()
        plt.xlabel("$\Delta r$ [meters]")
        plt.ylabel("$D_\Phi$(|$\Delta r$|) [rad$^2$]")
        plt.show()

def evaluate_PSD_accuracy(report_filename,
                          sim_result_directory,
                          min_max_freq = [1,100],
                          nsamples_psd = 250):
    """
    Evaluate the PSD of a series of phase screens from a turbulence evolved simulation

    :param report_filename: the filename including path for the output report of the turbulence simulation
    :param sim_result_directory: the directory in which the turbulence output simulation files over the timesteps have
                                 been written out
    :param min_max_freq: the minimum and maximum frequencies to include in the output plots
    :param nsamples_psd: the number of samples over which to bin when averaging the radial PSD
    :return: no return at the moment, just display plots
    """


    with open(report_filename, 'r') as f:
        sim_dict = yaml.load(f)

    # Get the necessary constants to define the simulation
    tsteps = sim_dict["Timesteps"]
    dx = sim_dict["Source dx"]
    N = sim_dict["N"]
    r0 = sim_dict["r0"]
    L0 = sim_dict['L0']
    l0 = sim_dict['l0']
    cn2 = float(sim_dict['Cn2'])
    wave = float(sim_dict['wavelength'])
    propdist = sim_dict['Propagation Dist']
    nscreens = sim_dict['Num Screens']


    # # Get the sidelength at the receiver and create a mask of the same size as the receiver pupil
    # side_len = N * dx
    # if D_receiver_pupil is None:
    #     D_receiver_pupil = side_len / 2
    # aperture_mask = MakePupil(D_receiver_pupil,
    #                           side_len,
    #                           N)

    # Theoretical PSD for the r0scrn value
    a = int(N / 2)
    nx = np.arange(0, a)
    deltaf = 1 / (N * dx)
    k = 2*np.pi / wave
    r0scrn = (0.423 * ((k) ** 2) * cn2 * (propdist / nscreens)) ** (-3.0 / 5.0)



    # Generate binning information for the empirical PSDs
    x = np.arange(-int(N / 2), int(N / 2))
    X, Y = np.meshgrid(x, x)
    R = np.sqrt((X * 2 * np.pi * deltaf) ** 2 + (Y * 2 * np.pi * deltaf) ** 2)
    R[int(N / 2), int(N / 2)] = 0
    rbin = (nsamples_psd * R / R.max()).astype(np.int)
    bin_sampling = (R.max()) * np.arange(0, rbin.max() + 1) / rbin.max()

    # Get the values within the bounds of interest
    f_sampling = bin_sampling / 2. / np.pi
    min_freq = min_max_freq[0]
    max_freq = min_max_freq[1]
    freq_bound_indices = np.logical_and(f_sampling > min_freq, f_sampling < max_freq)
    bin_sampling_bounds = bin_sampling[freq_bound_indices ]

    # For each simulation, calculate the structure function using the autocorrelation routine
    for scr_idx in range(0, nscreens):

        # Figure for plotting the structure function over time
        fig, ax = plt.subplots(figsize=(7,5))

        for t in range(0, tsteps):

            fname_scr_draw = sim_result_directory + "_scr" + str(scr_idx) + "_t" + "{:04d}".format(t) + ".npy"
            scr_draw = np.load(fname_scr_draw)

            # Old placeholder code that did not match theory: attempting to calculate the PSD using an autocorrelation approach
            # Calculate the phase screen autocorrelation function
            # U_r = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(scr_draw * aperture_mask))) * (dx) ** 2.0
            # U_r_autocorr = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(np.conj(U_r) * U_r))) * (deltaf * N) ** 2.0  # (deltaf * input_N) ** 2.0
            #
            # # Compensate for the effects of the masking at the receiver aperture
            # abs_fft_mask = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(aperture_mask))) * (dx) ** 2.0)
            # maskcorr = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(abs_fft_mask**2.0))) *  ((deltaf * N) ** 2.0)
            #                         # (input_N * deltaf) ** 2.0) * (input_dx) ** 2.0
            # B_r = np.abs(U_r_autocorr / maskcorr) * aperture_mask
            # PSD_empirical = (np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift((B_r)))) * (dx ) ** 2.0)**2) / N
            #
            # PSD_avg = scipy.ndimage.mean(PSD_empirical, labels=rbin, index=np.arange(0, rbin.max() + 1))
            # PSD_avg_bounds = PSD_avg[freq_bound_indices]

            # Scatter plot of the correlation function
            # ax.scatter(bin_sampling_bounds/2/np.pi, PSD_avg_bounds, c = 'b', alpha = 0.2)

            # Calculate the PSD of the screen using Ismail and Sathik (2015) equation
            # Approach:
            # 1. mean subtraction from phase screen
            # 2. Perform windowing of data to reduce leakage
            # 3. Take FFT and scale
            # 4. Squared value gives the PSD of the dataset at current timestep

            # Old version with incorrect scale factor
            # h = np.hamming(N)
            # ham2d = np.sqrt(np.outer(h, h))
            # window_gain =  (ham2d.sum()) / (N**2)
            # # Account for correlated gain of the window for noise power and the doubling of power for negative frequencies
            # scale_factor = 2 * window_gain
            # # fft_ph = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ham2d*(scr_draw - np.mean(scr_draw)))))* (dx) ** 2.0
            # fft_ph = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ham2d*(scr_draw ))))* (dx) ** 2.0
            # PSD_emp_2 = (np.abs(fft_ph)**2.0) / scale_factor #* scale_factor
            # PSD_avg_2 =  scipy.ndimage.mean(PSD_emp_2, labels=rbin, index=np.arange(0, rbin.max() + 1))
            # PSD_avg_2_bounds = PSD_avg_2[freq_bound_indices]
            # ax.scatter(bin_sampling_bounds/2/np.pi, PSD_avg_2_bounds, c = 'b', alpha = 0.2)

            # New version with scale factor that should be more accurate
            # h = np.hamming(N)
            # ham2d = np.outer(h, h)
            # window_gain =  np.sum(ham2d**2.0)/(N**2)
            # # Account for correlated gain of the window for noise power and the doubling of power for negative frequencies
            # scale_factor = 1 * window_gain
            # # fft_ph = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ham2d*(scr_draw - np.mean(scr_draw)))))* (dx) ** 2.0
            # fft_ph = np.fft.fftshift(np.fft.fft2(np.fft.fftshift((scr_draw * ham2d / N**2 )))) # *ham2d
            # PSD_emp_2 = (np.abs(fft_ph)**2.0) * scale_factor
            # PSD_avg_2 =  scipy.ndimage.mean(PSD_emp_2, labels=rbin, index=np.arange(0, rbin.max() + 1))
            # PSD_avg_2_bounds = PSD_avg_2[freq_bound_indices]
            # ax.scatter(bin_sampling_bounds/2/np.pi, PSD_avg_2_bounds, c = 'b', alpha = 0.2)

            # Factoring for signal rms
            h = np.hamming(N)
            ham2d = np.outer(h, h)
            noise_gain =  np.sum(ham2d**2.0)/(N**2)
            coherent_gain = np.sum(ham2d) / (N**2)
            fbin = 1/(N*dx)

            # Account for correlated gain of the window for noise power and the doubling of power for negative frequencies
            scale_factor = noise_gain * (fbin**2.0) /(coherent_gain**3.0)
            fft_ph = np.fft.fftshift(np.fft.fft2(np.fft.fftshift((scr_draw * ham2d / (N**2) ))))
            PSD_emp_2 = (np.abs(fft_ph)**2.0) / scale_factor
            PSD_avg_2 =  scipy.ndimage.mean(PSD_emp_2, labels=rbin, index=np.arange(0, rbin.max() + 1))
            PSD_avg_2_bounds = PSD_avg_2[freq_bound_indices]
            ax.scatter(bin_sampling_bounds/2/np.pi, PSD_avg_2_bounds, c = 'b', alpha = 0.2)
            # ax.scatter(R/2/np.pi, PSD_emp_2, c = 'b', alpha = 0.2)

            # fft_scr = np.fft.fftshift(np.fft.fft2(scr_draw))*dx**2.0
            # p_scr = np.abs(fft_scr*np.conj(fft_scr))#/(N**4)
            # p_scr_avg_2 = scipy.ndimage.mean(p_scr, labels=rbin, index=np.arange(0, rbin.max() + 1))
            # p_scr_avg_2_bounds = p_scr_avg_2[freq_bound_indices]
            # # ax.scatter(bin_sampling_bounds / 2 / np.pi, p_scr_avg_2_bounds, c='k', alpha=0.2)
            # # ax.scatter(R / 2 / np.pi, p_scr, c='k', alpha=0.2)





        # Theoretical PSD function for the Kolmogorov spectrum
        K0 =2*np.pi/L0
        PSD_theor = 0.49 * (r0scrn**(-5/3)) * (bin_sampling_bounds**2  + K0**2.0)**(-11/6)

        ax.plot(bin_sampling_bounds/2/np.pi, PSD_theor, 'red', label = "theory")
        # ax.plot(bin_sampling/2/np.pi, PSD_avg, 'b--')
        ax.set_xlim([min_freq,max_freq])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.set_title("Screen #" + str(scr_idx))
        ax.set_xlabel("spatial frequency [1/m]")
        ax.set_ylabel("PSD(f)")
        plt.show()




# report_fname = "H:\projects\wavepy\wavepy_v2_runs\collected_energy_sim\cn2_1e-13\\run1\simulation_params.txt"
# sim_dir = "H:\projects\wavepy\wavepy_v2_runs\collected_energy_sim\cn2_1e-13\\run1\\"
# # report_fname = "H:\projects\wavepy\wavepy_v2_runs\\directed_wind\cn2_1e-15\simulation_params.txt"
# # sim_dir = "H:\projects\wavepy\wavepy_v2_runs\\directed_wind\cn2_1e-15\\"
# structure_function_over_time(report_fname,
#                              sim_dir,
#                              D_receiver_pupil = None )#,  str_fcn_num_samples = 100)

# evaluate_phase_structure_function_accuracy(512, 0.0023, 1e-13, num_screen_draws=10, input_num_subharmonics=5)

# report_fname = "H:\projects\wavepy\wavepy_v2_runs\collected_energy_sim\cn2_1e-13\\run1\simulation_params.txt"
# sim_dir = "H:\projects\wavepy\wavepy_v2_runs\collected_energy_sim\cn2_1e-13\\run1\\"

# report_fname = "H:\projects\wavepy\wavepy_v2_runs\\random_wind\cn2_1e-14\subset\simulation_params.txt"
# sim_dir = "H:\projects\wavepy\wavepy_v2_runs\\random_wind\cn2_1e-14\subset\\"
# evaluate_phase_structure_function_accuracy_postprocess(report_fname,  sim_dir)

# report_fname = "H:\projects\wavepy\wavepy_v2_runs\directed_wind\cn2_1e-14\subset\simulation_params.txt"
# sim_dir = "H:\projects\wavepy\wavepy_v2_runs\directed_wind\cn2_1e-14\subset\\"
# evaluate_PSD_accuracy(report_fname,  sim_dir)