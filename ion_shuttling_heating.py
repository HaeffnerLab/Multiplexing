# Import standard libraries
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Numerical tools
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import fsolve, leastsq,root,brentq,newton,curve_fit
from sympy.utilities.lambdify import lambdify
from sympy import sympify
from sympy import var

import timeit
from time import time
from mpl_toolkits import mplot3d

# Set matplotlib parameters for consistent figure styling
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "xtick.labelsize": 12,
    "xtick.direction": "in",
    "xtick.major.pad": 3,
    "xtick.top": True,
    "ytick.labelsize": 12,
    "ytick.direction": "in",
    "ytick.right": True,
    "axes.labelsize": 12,
    "axes.labelpad": 3,
    "axes.grid": True,
})

# Constants
E0 = 8.85e-12
e = 1.6e-19
m = 6.64215627e-26
pi = math.pi
h_bar = 1.0546e-34

# Pre‑compute constant that appears inside the ODE so it is evaluated only once
coulomb_factor = e ** 2 / (4 * pi * E0)

# Parameters for the simulation
# T_factor = (121 / 133.93969396934) * 1.1625  # The time factor for the simulation
# voltage_change_factor = 1.00  # The factor to change the voltage
# NUM_VOLTAGE_POINTS = 500  # The points for voltage sampling
# T = 150 * 1e-6 * T_factor # Total time duration in s unit
# points = 3000000  # Number of points at which to evaluate the solution
# method = 'RK45'

def pfit(x_data, y_data, degree, plot=False):
    """
    Polynomial fit of data.

    Parameters:
    - x_data: x data points
    - y_data: y data points
    - degree: Degree of the polynomial
    - plot: Boolean to plot the original data and the fit

    Returns:
    - coeffs: Coefficients of the polynomial fit
    """
    coeffs = np.polyfit(x_data, y_data, degree)
    poly = np.poly1d(coeffs)
    y_fit = poly(x_data)
    # if plot:
    #     plt.plot(x_data, y_data, label='Original')
    #     plt.plot(x_data, y_fit, label='Fit')
    #     plt.legend()
    #     plt.show()
    return coeffs

def v_har(delta_x, x_data, y_data, plot=False):
    """
    Harmonic fit around a minimum point within a specified range.

    Parameters:
    - delta_x: Range around the minimum point to consider for the fit
    - x_data: x data points
    - y_data: y data points
    - plot: Boolean to plot the original data and the fit

    Returns:
    - coeffs: Coefficients of the polynomial fit
    """
    def mini(x_data, y_data):
        """Find the minimum point of the potential."""
        min_index = np.argmin(y_data)
        return x_data[min_index]

    min_x = mini(x_data, y_data)
    mask = (x_data >= min_x - delta_x) & (x_data <= min_x + delta_x)
    fit_x_data = x_data[mask]
    fit_y_data = y_data[mask]

    coeffs = np.polyfit(fit_x_data, fit_y_data, 2)  # Degree is always 2 for harmonic fit
    poly = np.poly1d(coeffs)
    y_fit = poly(x_data)

    # if plot:
    #     plt.plot(x_data, y_data, label='Original')
    #     plt.plot(x_data, y_fit, label='Fit')
    #     plt.xlabel('x/m')
    #     plt.ylabel('Potential Energy/J')
    #     plt.legend()
    #     plt.show()

    return coeffs

def scale_voltage_array(voltage_array, original_min, original_max, new_min, new_max):
    """Scales the voltage array based on the original and new min/max values."""
    return ((voltage_array - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min

def load_and_process_data(filepath, smooth = False, smooth_window = 100):
    """Loads the CSV, processes the data, and saves time and scaled voltage arrays."""
    # Load the CSV file
    df = pd.read_csv(filepath)
    if(smooth == True):
        df = df.groupby(np.arange(len(df)) // smooth_window).mean()

    # Extract arrays of time and voltage, apply scaling and corrections
    time_array = df['Time (us)'].to_numpy() * T_factor * 1e6
    voltage_array = df['Value'].to_numpy() * (-0.0682 * 1e6) + 20.17
    
    # Scale the first voltage array
    original_min, original_max = min(voltage_array), max(voltage_array)
    scaled_voltage_array = scale_voltage_array(voltage_array, original_min, original_max, 20.21, 25.59)
    
    # Save arrays
    np.save('time_array.npy', time_array)  # Unit: us
    np.save('DC1_voltage_array.npy', scaled_voltage_array)
    
    # Process for the second voltage array with different scaling
    voltage_array = -voltage_array
    original_min, original_max = min(voltage_array), max(voltage_array)
    scaled_voltage_array = scale_voltage_array(voltage_array, original_min, original_max, 23.33, 28.68)
    
    # Save the second voltage array
    np.save('DC2_voltage_array.npy', scaled_voltage_array)
    
    return time_array, scaled_voltage_array

def plot_voltage_time(time_array, scaled_voltage_array, pdf_filename):
    """Plots the voltage array over time and saves the plot to a PDF."""
    half_column_width_inches = 4.25
    aspect_ratio = 2
    # plt.figure(figsize=(half_column_width_inches, half_column_width_inches / aspect_ratio))
    # plt.title('Shuttling function', fontdict={'family': 'STIXGeneral', 'size': 12})
    # plt.plot(time_array, scaled_voltage_array, 'C0', linewidth=2.5, label='Programmed waveform')
    # plt.axvline(13 * T_factor, color='r', linewidth=0.5, linestyle='--')
    # plt.axvline((15) * T_factor, color='r', linewidth=0.5, linestyle='--')
    # plt.xlabel(r'Time ($\mu$$s$)', fontdict={'family': 'STIXGeneral', 'size': 12})
    # plt.ylabel('Voltage (V)', fontdict={'family': 'STIXGeneral', 'size': 12})
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    # plt.show()

def ion_shuttling_heating(T):
    points = 3000000  # Number of points at which to evaluate the solution
    filepath = 'shuttling_function.csv'
    # filepath = 'RigolDS0_123.csv'
    time_array, scaled_voltage_array = load_and_process_data(filepath, smooth = True, smooth_window = 10)
    # scaled_voltage_array = np.ones(len(scaled_voltage_array)) * scaled_voltage_array[0]

    # Displaying the first few elements to verify the extraction
    print(f"Length of time_array: {len(time_array)}, Length of scaled_voltage_array: {len(scaled_voltage_array)}")
    print(f"scaled_voltage_array_min = {min(scaled_voltage_array)}, scaled_voltage_array_max = {max(scaled_voltage_array)}")
    print("First 5 elements of time_array:", time_array[:5])
    print("First 5 elements of scaled_voltage_array:", scaled_voltage_array[:5])
    print("The time range = ", time_array[-1], " us")

    # Plotting preparation (optional, kept but commented to avoid I/O overhead)
    time_array = np.load("time_array.npy") * 1e-6
    DC1_voltage_array = np.load("DC1_voltage_array.npy") * voltage_change_factor
    DC2_voltage_array = np.load("DC2_voltage_array.npy") * voltage_change_factor

    # plot_voltage_time(time_array * 1e6, scaled_voltage_array, 'Shuttling_function_new.pdf')

    # ---------------------------------------------------------------------
    # 1. Pre‑compute ω(t) and x₀(t) on the waveform grid and build
    #    cubic‑spline interpolants so that the ODE does not need to call the
    #    expensive harmonic fit at every RHS evaluation.
    # ---------------------------------------------------------------------

    half_column_width_inches = 4.25  # kept for potential plotting later
    aspect_ratio = 2

    # Load electrode potential files once
    data_dc1 = np.loadtxt('/Users/bingran_you/Documents/GitHub_MacBook/Multiplexing/ion-shuttling/DC1.csv',
                          delimiter=',', skiprows=9)
    data_dc2 = np.loadtxt('/Users/bingran_you/Documents/GitHub_MacBook/Multiplexing/ion-shuttling/DC2.csv',
                          delimiter=',', skiprows=9)

    def cal_omega(DC1, DC2, delta_x, plot):
        """Return (ω, x₀) for given electrode scalings (no plotting)."""
        V1 = e * DC1 * data_dc1[:, 3]
        V2 = e * DC2 * data_dc2[:, 3]
        V  = V1 + V2
        z  = 1e-3 * data_dc1[:, 2]
        a, b, _ = v_har(delta_x, z, V, plot)
        omega = np.sqrt(2 * a / m)
        x_0   = -b / (m * omega**2)
        return omega * 1.10, x_0

    # Vectorised pre‑computation on the full voltage trace
    omega_values = np.empty_like(time_array)
    x0_values    = np.empty_like(time_array)
    for idx, (v1, v2) in enumerate(zip(DC1_voltage_array, DC2_voltage_array)):
        omega_values[idx], x0_values[idx] = cal_omega(v1, v2, 1e-4, False)

    # Build fast cubic splines for use inside the ODE (extrapolate outside range)
    # Use zero‑order hold (nearest) to match original piece‑wise constant
    omega_of_t = interp1d(time_array, omega_values, kind='nearest', fill_value="extrapolate")
    x0_of_t    = interp1d(time_array, x0_values,    kind='nearest', fill_value="extrapolate")

    # ---------------------------------------------------------------------
    # 2. Right‑hand side of the differential equations – now fully vectorised
    # ---------------------------------------------------------------------

    def ode_csv(t, y):
        # Split positions and velocities
        pos = y[:9]
        vel = y[9:]

        # Trap parameters at this time (interpolated)
        omega_val = float(omega_of_t(t))
        x0_val    = float(x0_of_t(t))

        # Harmonic part a x + b
        a = -m * omega_val**2
        b = m * omega_val**2 * x0_val

        # Vectorised Coulomb term
        diff = pos[:, None] - pos[None, :]
        np.fill_diagonal(diff, np.inf)  # avoid zero division on the diagonal
        # Match the original sign convention: coefficient = -sign(diff)/(diff^2)
        coulomb_sum = coulomb_factor * np.sum(-np.sign(diff) / diff**2, axis=1)

        accel = (a * pos + b - coulomb_sum) / m

        # Concatenate derivatives
        dydt = np.empty_like(y)
        dydt[:9] = vel
        dydt[9:] = accel
        return dydt

    def Eqposition3(N):
        # Define the equations to solve
        def func(u):
            return np.array([u[m] - np.sum(1 / (u[m] - u[:m])**2) + np.sum(1 / (u[m] - u[m+1:])**2) for m in range(N)])
        
        # Define the first-order derivative
        def dfunc(u):
            return np.array([1 + 2 * np.sum(1 / (u[m] - u[:m])**3) - 2 * np.sum(1 / (u[m] - u[m+1:])**3) for m in range(N)])
        
        # Define the second-order derivative (not used in newton method here but defined for completeness)
        def ddfunc(u):
            return np.array([np.sum(1 / (u[m] - u[:m])**4) - np.sum(1 / (u[m] - u[m+1:])**4) for m in range(N)])

        # Initial guess for the Newton method
        ni = np.arange(0, N)
        guess = 3.94 * (N**0.387) * np.sin(1 / 3 * np.arcsin(1.75 * N**(-0.982) * ((ni + 1) - (N + 1) / 2)))
        
        # Solve for the equilibrium positions using the Newton method
        x0 = newton(func, guess, fprime=dfunc, maxiter=100000)
        
        return np.round(x0, 5)

    # Initial conditions setup
    omega_value, x0_value = cal_omega(DC1_voltage_array[0], DC2_voltage_array[0], 1e-4, False)
    # print(omega_value, x0_value)
    x0 = x0_value
    print("The initial position: ", x0_value)
    omega0 = omega_value
    print("The trap frequency: ", omega0)
    l0 = math.pow(e**2 / (4 * math.pi * E0 * m * omega0**2), 1/3)

    # Calculate initial positions and scale
    initial_positions = Eqposition3(9) * l0
    initial_state = [x + x0 for x in initial_positions] + [0] * 9  # Append velocity initial conditions (zeros)

    # Convert units to micrometers (um) and back, demonstrating unit conversion without functional change
    initial_state_um = [x * 1e6 for x in initial_state]
    print(initial_state_um)  # In the unit of micrometers (um)
    initial_state = [x / 1e6 for x in initial_state_um]  # Convert back, though this step is redundant here
    # initial_state = np.array([17.08369540791233, 28.474782827470513, 38.13198316843803, 47.06403315776806, 55.70326488240171, 64.34249658649416, 73.274546588792, 82.93174692487013, 94.32283434541237, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / 1e6

    # initial_state = np.array([17.083695539177068, 28.47478293928541, 38.131983453958746, 47.06403296135834, 55.70326497471302, 64.34249664063852, 73.27454636648979, 82.93174679858147, 94.3228342153568, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / 1e6

    # initial_state = np.array([17.08369553804403, 28.474782951113287, 38.131983421221584, 47.06403299876212, 55.70326497124225, 64.3424966085964, 73.27454639684315, 82.93174678729447, 94.32283421644193, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / 1e6

    # initial_state = np.array([17.083695539177068, 28.47478293928541, 38.13198342382467, 47.06403296135834, 55.70326497634435, 64.34249664063852, 73.27454636648979, 82.93174679858147, 94.3228342153568, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / 1e6

    # Integrate with adaptive step size and dense output.  We will sample
    # the solution afterwards instead of forcing the solver to hit every
    # point on a huge uniform grid.
    solution = solve_ivp(ode_csv, (0, T), initial_state, method=method,
                         atol=1e-10, rtol=1e-8, dense_output=True)

    # Choose a much lighter sampling grid for post‑processing.
    sample_points = 300_000  # adjust if finer output is required
    time_points = np.linspace(0, T, sample_points)
    positions = solution.sol(time_points)[:9]

    # Time windows
    start_time_array = np.array([0, 13e-6, 25e-6, 37e-6, 49e-6, 61e-6, 73e-6, 85e-6, 97e-6, 140e-6]) * 1e6 * T_factor
    stop_time_array = np.array([2e-6, 15e-6, 27e-6, 39e-6, 51e-6, 63e-6, 75e-6, 87e-6, 99e-6, 150e-6]) * 1e6 * T_factor

    # for i in range(9):
    #     if(True):
    #         plt.plot(time_points * 1e6, (positions[i] - positions[i][0]) * 1e6, lw=2, label=f'ion {i}')

    for i in range(9):
        print(np.average(positions[i]) * 1e6)

    # print([17.08366553858873, 28.47476582463269, 38.132043127702964, 47.06397638554019, 55.703264876617695, 64.3425533676952, 73.27448662553242, 82.9317639286027, 94.32286421464666])

    # Calculate amplitude in these time windows
    time_points_us = time_points * 1e6  # Convert time_points to microseconds
    amplitudes = []
    # start_time = time_points_us[-15000]
    start_time = time_points_us[0]
    stop_time = time_points_us[-1]

    def nor_cor9(positions_minus_x0_values):
        # Assuming positions_minus_x0_values is a 2D array where each column corresponds to x1 through x9
        
        # Coefficients for each y computation as rows in a matrix
        coefficients = np.array([
            [0.333384, 0.333368, 0.333329, 0.333377, 0.333383, 0.333377, 0.333329, 0.333368, 0.333384],
            [-0.533933, -0.376401, -0.24288, -0.119431, 6.89694e-19, 0.119431, 0.24288, 0.376401, 0.533933],
            [-0.553209, -0.096929, 0.165769, 0.30783, 0.353079, 0.30783, 0.165769, -0.096929, -0.553209],
            [-0.439418, 0.282817, 0.401852, 0.255804, -9.57488e-17, -0.255804, -0.401852, -0.282817, 0.439418],
            [0.281185, -0.510756, -0.187306, 0.22284, 0.388073, 0.22284, -0.187306, -0.510756, 0.281185],
            [0.146521, -0.501459, 0.258135, 0.400506, -9.81546e-17, -0.400506, -0.258135, 0.501459, -0.146521],
            [0.0613199, -0.340712, 0.527405, -0.0227322, -0.450561, -0.0227322, 0.527405, -0.340712, 0.0613199],
            [-0.0196748, 0.163883, -0.461312, 0.509813, 3.83695e-18, -0.509813, 0.461312, -0.163883, 0.0196748],
            [-0.0042412, 0.0501759, -0.219504, 0.493967, -0.640796, 0.493967, -0.219504, 0.0501759, -0.0042412]
        ])

        # Dot product of coefficients with positions_minus_x0_values to compute all y values at once
        positions_modes = np.dot(coefficients, positions_minus_x0_values)

        return positions_modes

    print(len(positions[1]))

    # 'time_points' and 'positions' are already defined from sampling the
    # dense solution above.  Avoid overwriting them with the sparse solver
    # output.
    # Interpolated ω(t) and x₀(t) on the same sampling grid
    omega_values_sampled = omega_of_t(time_points)
    x0_values_sampled    = x0_of_t(time_points)

    positions_um = positions * 1e6
    x0_value_array_um = np.array(x0_values_sampled) * 1e6

    # Subtract x0 from each ion position before mode projection
    pos_minus_x0 = positions - x0_values_sampled
    positions_modes = nor_cor9(pos_minus_x0) / 3

    factor = 0.34 * 1.0004196441454332
    for i, pos in enumerate(positions_modes, start=1):
        # plt.scatter(time_points * 1e6, pos * 1e6, s=0.2, label=f'Mode {i}')
        if(i != 2):
            if(i == 8):
                # plt.plot(time_points * 1e6, np.abs(pos - pos[0]) * 1e6 / factor, lw=1, c="C1", label=f'Mode {i}')
                continue
            if(i == 1):
                # plt.plot(time_points * 1e6, np.abs(pos - pos[0]) * 1e6 / factor, lw=1, c="black", alpha = 0.7, label=f'Mode {i}')
                print(np.max(np.abs(pos - pos[0]) * 1e6 / factor))
                continue
            # plt.plot(time_points * 1e6, np.abs(pos - pos[0]) * 1e6 / factor, lw=1, label=f'Mode {i}')
        else:
            # plt.plot(time_points * 1e6, np.abs(pos - pos[0]) * 1e6 / factor, lw=1, label=f'Mode {i}')
            continue

    # Calculate amplitude in these time windows
    time_points_us = time_points * 1e6  # Convert time_points to microseconds

    # ------------------------------------------------------------------
    #  Calculate amplitudes only in the final 10 % of the simulation, to
    #  match the original code behaviour (it previously used the last
    #  300 000 samples out of 3 000 000).
    # ------------------------------------------------------------------

    tail_fraction = 0.10
    start_time = time_points_us[int((1 - tail_fraction) * len(time_points_us))]
    stop_time  = time_points_us[-1]

    amplitudes = []
    for i, pos in enumerate(positions_modes, start=1):
        mask = (time_points_us >= start_time) & (time_points_us <= stop_time)
        amplitude = (pos * 1e6)[mask].max() - (pos * 1e6)[mask].min()
        amplitudes.append(amplitude)
    print(amplitudes)
    quantum = h_bar * omega_value
    real_amplitudes = np.array(amplitudes) * 1e-6
    n_bar = [0.25 * m * omega_value**2 * amplitude**2 / quantum for amplitude in real_amplitudes]

    return n_bar

# list = points = np.linspace(1.25, 1.7, 2)
list=points=[1.25]
results = []
total_results = []

for i in range(len(list)):
    # Parameters for the simulation
    method = 'RK45'
    voltage_change_factor = 1.00  # The factor to change the voltage
    T_factor = (121 / 133.93969396934) * list[i]  # The time factor for the simulation
    NUM_VOLTAGE_POINTS = 500  # The points for voltage sampling
    T = 150 * 1e-6 * T_factor # Total time duration in s unit
    n_bar = ion_shuttling_heating(T)
    print(n_bar)
    results.append(n_bar[0])
    total_results.append(n_bar)
    print(f"Current list of shuttling time in us (T): {T} with heating {n_bar[0]} quanta")
    print("\nCurrent final results: ", results)
print(results)
