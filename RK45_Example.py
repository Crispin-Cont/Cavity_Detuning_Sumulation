import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

class PIDController:
    """A simple PID controller class."""
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self._integral = 0
        self._last_error = 0

    def update(self, current_value, dt):
        """
        Calculates and returns the control signal.
        Args:
            current_value (float): The current process variable (e.g., total_detuning).
            dt (float): The time step.
        Returns:
            float: The control signal output.
        """
        error = self.setpoint - current_value
        
        # Proportional term
        proportional = self.Kp * error
        
        # Integral term
        self._integral += error * dt
        integral = self.Ki * self._integral
        
        # Derivative term
        derivative = self.Kd * ((error - self._last_error) / dt)
        self._last_error = error
        
        # Calculate and return the control signal
        control_signal = proportional + integral + derivative
        return control_signal


class CavitySimulator:
    def __init__(self, tmax=1.0, dt=1e-3, **kwargs):
        """Initializes the simulator with physical and simulation parameters."""
        # --- Simulation Parameters ---
        self.tmax = tmax
        self.dt = dt

        # --- Physical Parameters ---
        self.O = kwargs.get('O', 2 * np.pi * np.array([157, 182, 189, 215, 292, 331, 380, 412, 462, 471]))
        self.tau_mech = kwargs.get('tau_mech', 2 * np.array([56.8, 113.2, 70.38, 25, 300, 304.2, 409.46, 305.11, 202.19, 205.86]) * 1e-3)
        # Note: kLFD value is in Hz/(MV/m)^2, so factor 1e-6 is included
        self.kLFD = kwargs.get('kLFD', 2 * np.pi * 2.9636 * np.array([0.02, 0.03, 0.055, 0.55, 0.085, 0.029, 0.052, 0.19, 0.075, 0.095]) * 1e-6)
        self.kpiezo = kwargs.get('kpiezo', 2 * np.pi * 16.9348 * np.array([0.02, 0.03, 0.055, 0.55, 0.085, 0.029, 0.052, 0.19, 0.075, 0.095]))
        self.kmicro = kwargs.get('kmicro', 2 * np.pi * 64.0982 * np.array([0.02, 0.03, 0.055, 0.55, 0.085, 0.029, 0.052, 0.19, 0.075, 0.095]))
        
        # Number of mechanical modes
        self.NUM_MODES = len(self.O)

        # Cavity parameters
        self.Qext = 3e6
        self.Q0 = 2.7e10
        self.beta = self.Q0 / self.Qext
        self.QL = self.Q0 / (1 + self.beta)
        self.w0 = 2 * np.pi * 650e6
        self.whalf = self.w0 / (2 * self.QL)
        self.Leff = 1.038

        # Initial State Vector: [Dw1..DwN, dDwdt1..dDwdtN, v_real, v_imag]
        self.y0 = np.concatenate([np.zeros(2 * self.NUM_MODES), [25.0, 0.0]])
        
        # Mutable variable for the control signal
        self.piezo_control = [0.0]

    def _V_forward_drive(self, t):
        # ... (same as before)
        tau = 2 * self.QL / self.w0
        Leff = self.Leff
        t1 = 1e-3
        tfill = t1 + tau * np.log(2)
        tflat = tfill + t1
        ratio = (1 - np.exp(-(tfill - t1) / tau))
        Amp = (25e6 * Leff)
        drive = Amp * ((t >= t1) * (t < tfill) + ratio * (t >= tfill) * (t < tflat))
        return drive

    def _microphonics_drive(self, t):
        # ... (same as before)
        return 20 * np.cos(2 * np.pi * 20 * t)

    def _coupled_dynamics(self, t, y):
        """
        The dynamics function that now uses the external piezo control signal.
        """
        Dw_array = y[0:self.NUM_MODES]
        dDwdt_array = y[self.NUM_MODES:2*self.NUM_MODES]
        v_real = y[2*self.NUM_MODES]
        v_imag = y[2*self.NUM_MODES + 1]
        
        v_complex = v_real + 1j * v_imag
        
        # Use the control signal from the main loop
        piezo = self.piezo_control[0]
        microphonics = self._microphonics_drive(t)
        V_forward = self._V_forward_drive(t)
        
        d_Dw_dt_array = dDwdt_array
        
        D2wdt2_array = - (2 / self.tau_mech) * dDwdt_array - self.O**2 * Dw_array + self.kLFD * np.abs(v_complex / self.Leff)**2 + self.kpiezo * piezo + self.kmicro * microphonics
        
        detuning_total = np.sum(Dw_array)
        
        dvdt_complex = (-self.whalf + detuning_total * 1j) * v_complex + 2 * self.whalf * V_forward
        
        dv_real_dt = dvdt_complex.real
        dv_imag_dt = dvdt_complex.imag
        
        return np.concatenate([d_Dw_dt_array, D2wdt2_array, [dv_real_dt, dv_imag_dt]])
        
    def plot_results(self, t_sol, detuning_total_sol, v_mag_sol, piezo_control_sol):
        """Generates plots for the simulation results from provided data."""
        plt.figure(figsize=(12, 12))

        plt.subplot(4, 1, 1)
        plt.plot(t_sol, detuning_total_sol, label='Total Detuning')
        plt.title('Total Mechanical Detuning')
        plt.xlabel('Time (s)')
        plt.ylabel('Detuning ($\Delta\omega$)')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(t_sol, v_mag_sol, label='Cavity Voltage Magnitude')
        plt.title('Cavity Voltage')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage Magnitude (|V|)')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(4, 1, 3)
        plt.plot(t_sol, piezo_control_sol, label='Piezo Control Signal')
        plt.title('PID Control Output')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Signal (V)')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(t_sol, self._V_forward_drive(t_sol), label='Forward Voltage')
        plt.title('Forward Voltage Drive')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# --- Main execution block with the control loop ---
if __name__ == "__main__":
    
    # 1. Instantiate the simulator
    simulator = CavitySimulator()
    
    # 2. Instantiate the PID controller with a target detuning of 0
    # Kp=1, Ki=10, Kd=0.1 are example gains; these would need to be tuned.
    pid_controller = PIDController(Kp=1.0, Ki=10.0, Kd=0.1, setpoint=0.0)

    # 3. Initialize the RK45 solver
    solver = RK45(
        fun=simulator._coupled_dynamics,
        t0=0,
        y0=simulator.y0,
        t_bound=simulator.tmax,
        max_step=simulator.dt
    )

    # Lists to store the results
    t_list = []
    detuning_list = []
    v_mag_list = []
    piezo_control_list = []

    # 4. Main control loop
    while solver.status == 'running':
        solver.step()
        
        # Get the current time and state vector
        t_current = solver.t
        y_current = solver.y

        # Unpack the detuning array to get the total detuning
        Dw_array = y_current[0:simulator.NUM_MODES]
        total_detuning = np.sum(Dw_array)
        
        # 5. Get the control signal from the PID loop
        dt_step = solver.t - t_list[-1] if t_list else solver.t
        control_signal = pid_controller.update(total_detuning, dt_step)
        
        # 6. Apply the control signal back to the simulator for the next step
        simulator.piezo_control[0] = control_signal
        
        # 7. Calculate and store the values for plotting
        v_real = y_current[2 * simulator.NUM_MODES]
        v_imag = y_current[2 * simulator.NUM_MODES + 1]
        v_mag = np.sqrt(v_real**2 + v_imag**2)

        t_list.append(t_current)
        detuning_list.append(total_detuning)
        v_mag_list.append(v_mag)
        piezo_control_list.append(control_signal)
    
    # 8. Plot the results collected from the loop
    simulator.plot_results(np.array(t_list), np.array(detuning_list), np.array(v_mag_list), np.array(piezo_control_list))