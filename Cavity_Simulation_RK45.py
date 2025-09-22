import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
import time

start_time=time.time() # record the start time
# --- Time and Simulation Parameters ---
tmax = 1  # Max time in seconds
dt = 1e-3   # Time step in seconds

# --- Time-dependent drive functions ---
def V_forward_drive(t):
    """Calculates the forward voltage at a given time t."""
    Qext = 3e6
    Q0 = 2.7e10
    beta = Q0 / Qext
    QL = Q0 / (1 + beta)
    w0 = 2 * np.pi * 650e6  # Hz
    tau = 2 * QL / w0       # seconds
    Leff = 1.038            # Meters

    t1 = 1e-3
    tfill = t1 + tau * np.log(2)
    tflat = tfill + t1
    ratio = (1 - np.exp(-(tfill - t1) / tau))
    Amp = (25 * Leff)     # V/m * m = V
    drive = Amp*0.5*(t >= 0.0) #CW
    #drive = Amp * ((t >= t1) * (t < tfill) + ratio * (t >= tfill) * (t < tflat)) #pulse operation 
    return drive



mu = 1e-9
eta = 1e-9
gamma = 0.01
I_mk = np.array([0, 0, 0, 0])
Q_mk = np.array([0, 0, 0, 0])  
phi_mk  = np.array([0, 0, 0, 0]) 
omega_m = 2*np.pi*np.array([5, 10 ,20 ,50])

def piezo_drive(t,detuning):
    """Piezo drive changes over time."""
    
    I_m = I_mk
    Q_m = Q_mk
    I_m += -mu*detuning*np.cos(omega_m*t-phi_mk) #calculates the k+1
    Q_m += -mu*detuning*np.sin(omega_m*t-phi_mk) #calculates the k+1   
    phi_m += -eta*detuning*(I_mk*np.sin(omega_m*t-phi_m)-Q_mk*np.cos(omega_m*t-phi_m)) #calculates the k+1

    #Individual piezo driving terms
    piezo_m = I_mk *np.cos(omega_m*t) + Q_mk*np.sin(omega_m*t)

    # Calculate total output, this is based on the input from the user
    #output = proportional + integral + derivative
    output = np.sum(piezo_m)

    # Store for next iteration
    I_mk = I_m
    Q_mk = Q_m
    phi_mk = phi_m
    
    return output

wfreq_comp = 2*np.pi*np.array([5.0, 10.0, 20.0, 50.0])
def microphonics_drive(t):
    """Microphonics drive changes over time."""
    microphonics_slow = 0.01 * np.sin(2 * np.pi * 0.5 * t) 
    microphonics_sinousoid = np.sum( 0.05 * np.sin(wfreq_comp * t))
    microphonics = microphonics_slow + microphonics_sinousoid
    return microphonics


# --- System Parameters ---
# These parameter arrays define the number of mechanical modes
relative_amp = np.array([0.02, 0.03, 0.055, 0.55, 0.085, 0.029, 0.052, 0.19, 0.075, 0.095])
O = 2 * np.pi * np.array([157, 182, 189, 215, 292, 331, 380, 412, 462, 471])
tau_mech = 2 * np.array([56.8, 113.2, 70.38, 25, 300, 304.2, 409.46, 305.11, 202.19, 205.86]) * 1e-3  # seconds
kLFD = 2 * np.pi * 2.9636 * relative_amp
kpiezo = 2 * np.pi * 16.9348 * relative_amp
kmicro = 2 * np.pi * 64.0982 * relative_amp

# --- Determine the number of mechanical modes from the array sizes ---
NUM_MODES = len(O)

# Cavity parameters
Qext = 3e6
Q0 = 2.7e10
beta = Q0 / Qext
QL = Q0 / (1 + beta)
w0 = 2 * np.pi * 650e6  # Units in Hz
whalf = w0 / (2 * QL)
Leff = 1.038  # Units in Meters


def cavity_dynamics(t, y):
    """
    Defines the system of first-order ODEs for the coupled system.
    The state vector y is [Dw1...DwN, dDwdt1...dDwdtN, v_real, v_imag].
    """
    # Unpack the state vector based on the number of modes
    Dw_array = y[0:NUM_MODES]
    dDwdt_array = y[NUM_MODES:2*NUM_MODES]
    v_real = y[2*NUM_MODES]
    v_imag = y[2*NUM_MODES + 1]
    
    v_complex = v_real + 1j * v_imag
    detuning_total = np.sum(Dw_array)
    
    piezo = piezo_drive(t,detuning_total)
    microphonics = microphonics_drive(t)
    V_forward = V_forward_drive(t)
    
    # --- Define the derivatives ---
    d_Dw_dt_array = dDwdt_array
    
    O_squared = O**2
    
    
    dvdt_complex = (-whalf + detuning_total * 1j) * v_complex + 2 * whalf * V_forward
    
    D2wdt2_array = - (2 / tau_mech) * dDwdt_array - O_squared * Dw_array -O_squared * kLFD * np.abs((v_complex-25.95) / Leff)**2 + O_squared * kpiezo * piezo + O_squared * kmicro * microphonics    
    
    dv_real_dt = dvdt_complex.real
    dv_imag_dt = dvdt_complex.imag
    
    # Combine all derivatives into a single flat array to return
    return np.concatenate([d_Dw_dt_array, D2wdt2_array, [dv_real_dt, dv_imag_dt]])


# --- Main Simulation ---
# Dynamically create the initial state vector based on the number of modes
y0 = np.concatenate([np.zeros(2 * NUM_MODES), [25.95, 0.0]])
t_span = (0, tmax)

# Create and run the RK45 solver
solver = RK45(
    fun=cavity_dynamics,
    t0=t_span[0],
    y0=y0,
    rtol=1e-6,
    t_bound=t_span[1])

# Lists to store the solution points
t_sol = []
y_sol = []

while solver.status == 'running':
    t_sol.append(solver.t)
    y_sol.append(solver.y)
    solver.step()

# Append the final point
t_sol.append(solver.t)
y_sol.append(solver.y)

# Convert lists to NumPy arrays
t_sol = np.array(t_sol)
y_sol = np.array(y_sol).T

# Dynamically unpack the solution based on the number of modes
Dw_sol_array = y_sol[0:NUM_MODES]
dDwdt_sol_array = y_sol[NUM_MODES:2*NUM_MODES]
v_real_sol = y_sol[2*NUM_MODES]
v_imag_sol = y_sol[2*NUM_MODES + 1]

# Calculate the total detuning and voltage magnitude for plotting
detuning_total_sol = np.sum(Dw_sol_array, axis=0)
v_mag_sol = np.sqrt(v_real_sol**2 + v_imag_sol**2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time : {elapsed_time} s")


# --- Plot the Results ---
plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.plot(t_sol, detuning_total_sol/(2*np.pi), label='Total Detuning, $\sum \Delta\omega_i$')
plt.title('Total Mechanical Detuning')
plt.xlabel('Time (s)')
plt.ylabel('Detuning ($\Delta\omega$)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_sol, v_mag_sol, label='Cavity Voltage Magnitude, $|V|$')
plt.title('Cavity Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage Magnitude (|V|)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_sol, V_forward_drive(t_sol), label='Forward Voltage, $V_F$')
plt.title('Forward Voltage Drive')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()