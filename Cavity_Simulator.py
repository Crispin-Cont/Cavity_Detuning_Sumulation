import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional, List, Tuple, Dict, Any, Union
from pathlib import Path

class cavity_simulation(gym.Env):
    """
    Cavity simulation as a Gymnasium environment.
    
    Simulates N mechanical modes on the cavity. Only one electromagnetic mode is 
    considered for the cavity, the pi mode. The beam is not turned on. 

    The simulation can work for pulse operation and CW by maintaining the 
    forward power constant. 
    """
    
    def __init__(self, N: int = 3, Leff : float=1.01 , k_LFD = None, k_piezo = None, k_micro = None, 
                w_half: float = 2*np.pi*20, tau_mode = None, angular_mech_w = None, wfreq_comp = None, dt: float = 0.01, 
                max_piezo: float = 120, max_episode_steps: int = 1000, observation_noise_std: float = 0.0, operation: str = "pulsed",
                config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the coupled oscillators system as a Gymnasium environment.
        
        Parameters:
        -----------
        N : int
            Number of mechanical modes 
        Leff : floata
            The effective length for acceleration in a cavity, units of m
        k_LFD, k_piezo, k_micro : float, np.ndarray, or None
            Coupling constants. Can be:
            - float: uniform coupling 
            - np.ndarray (N,): full coupling array 
            - None: defaults to 0 
            - k_LFD  is the Lorentz forced detuning coefficient per mode, units of 2*pi*Hz/(MV/m)^2.
            - k_piezo  is the piezo coupling coefficient per mode, units of 2*pi*Hz/V. The voltage can be transferred
              into linear displacement of the piezo. 100 V is ~ 6 um at 2 K. 
            - k_micro this is the coupling to the microphonics. This value has never been measured but a general 
              value will be used. For the moment it will be 2*pi*Hz/Torr
        w_half : float
            This is the half bandwidth of the cavity 2*pi*f_Half of the cavity 
            electricmagnetic mode. The bandwidth of the cavity is determined by the beam current.  
        tau_mode :   float, np.ndarray, or None
            This is the time constant of the mechanical vibrations. Units are in s. 
            - float: uniform time constant 
            - np.ndarray (N,): full time constant array 
            - None: defaults to 1
        angular_mech_w : loat, np.ndarray, or None
            This is the mechanical modes of the cavity in angular frequencies. Units 2 *pi* Hz 
            - float: uniform angular frequencies
            - np.ndarray (N,): full angular mechanical frequencies array 
            - None: defaults to 1
        dt : float
            Time step for numerical integration
        max_piezo : float
            This is the maximum voltage that can be applied to the piezo. +/- 120 V.
        max_episode_steps : int
            Maximum number of steps per episode
        observation_noise_std : float
            Standard deviation of Gaussian noise added to observations (default: 0.0 = no noise)
        """
        super().__init__()
        
        # Load configuration from file if provided
        if config_file is not None:
            try:
                from config.config import ExperimentConfig
            except ImportError:
                import sys
                sys.path.append('../')
                from config.config import ExperimentConfig
            
            config = ExperimentConfig(config_file)
            sim_config = config.get_simulation_config()
            
            # Override parameters with values from config file
            N = sim_config.get('N', N)
            Leff = sim_config.get('Leff', Leff)
            k_LFD = sim_config.get('k_LFD', k_LFD)
            k_piezo = sim_config.get('k_piezo', k_piezo)
            k_micro = sim_config.get('k_micro', k_micro)
            w_half = sim_config.get('w_half', w_half)
            tau_mode = sim_config.get('tau_mode', tau_mode)
            angular_mech_w = sim_config.get('angular_mech_w', angular_mech_w)
            dt = sim_config.get('dt', dt)
            max_piezo = sim_config.get('max_piezo', max_piezo)
            max_episode_steps = sim_config.get('max_episode_steps', max_episode_steps)
            observation_noise_std = sim_config.get('observation_noise_std', observation_noise_std)
            
            # Store derived parameters for internal force calculations
            self.RL = sim_config.get('RL')
            self.Amp = sim_config.get('Amp') 
            self.t1 = sim_config.get('t1')
            self.tfill = sim_config.get('tfill')
            self.tflat = sim_config.get('tflat')
            self.ratio = sim_config.get('ratio')
            self.repetition_period = 1.0 / sim_config.get('repetition_rate', 20)  # Convert Hz to period
            

        self.N = N
        self.w_half= w_half
        self.Leff = Leff
        self.wfreq_comp = wfreq_comp
        self.operation = operation
        self.dt = dt
        self.max_piezo = max_piezo
        self.max_episode_steps = max_episode_steps
        
        # Observation noise parameters
        self.observation_noise_std = observation_noise_std

        # Store derived parameters for internal force calculations
        # Set defaults if no config file is provided
        if config_file is None:
            # Calculate basic defaults - these won't be realistic but allow basic operation
            # For realistic operation, use a configuration file
            Q0_default = 2.7e10
            Qext_default = 3e6
            beta_default = Q0_default / Qext_default
            QL_default = Q0_default / (1 + beta_default)
            RQ_default = 341
            RL_default = (0.5 * RQ_default) * QL_default
            accelerating_gradient_default = 25e6
            Amp_default = (accelerating_gradient_default * Leff) / RL_default
            tau_default = 2 * QL_default / (2 * np.pi * 650e6)
            
            self.RL = RL_default
            self.Amp = Amp_default
            self.t1 = 1e-3
            self.tfill = self.t1 + tau_default * np.log(2)
            self.tflat = self.tfill + self.t1
            self.ratio = (1 - np.exp(-(self.tfill - self.t1) / tau_default))
            self.repetition_period = 0.05  # 20 Hz
        
        self.microphonics_amplitude = 0.1  # Default microphonics amplitude
        self.microphonics_frequency = 0.5  # Default microphonics frequency (Hz)
        
        # Setup angular mechanical frequencies
        if angular_mech_w is None:
            #Default is 1
            self.angular_mech_w = np.ones(N, dtype=np.float32)
        elif isinstance(angular_mech_w, (int, float)):
            self.angular_mech_w = np.full(N, float(angular_mech_w), dtype=np.float32)
        else:
            self.angular_mech_w = np.array(angular_mech_w, dtype=np.float32)
            if len(self.angular_mech_w) != N:
                raise ValueError(f"angular mechanical frequencies array length {len(self.angular_mech_w)} must equal N={N}")

        # Setup coupling array for the LFD coefficient.
        if k_LFD is None:
            # Default: adjacent coupling with k=0
            self.k_LFD = np.zeros(N, dtype=np.float32)
        elif isinstance(k_LFD, (int, float)):
                self.k_LFD = np.full(N, float(k_LFD), dtype=np.float32)
        else:
            # Full coupling matrix
            self.k_LFD = np.array(k_LFD, dtype=np.float32)
            if len(self.k_LFD) != N:
                raise ValueError(f"k_LFD array length {len(self.k_LFD)} must be N= {N}")
            

        # Setup coupling array for the Piezo coefficient.
        if k_piezo is None:
            # Default: adjacent coupling with k=0
            self.k_piezo = np.zeros(N, dtype=np.float32)
        elif isinstance(k_piezo, (int, float)):
                self.k_piezo = np.full(N, float(k_piezo), dtype=np.float32)
        else:
            # Full coupling matrix
            self.k_piezo = np.array(k_piezo, dtype=np.float32)
            if len(self.k_piezo) != N:
                raise ValueError(f"k_piezo array length {len(self.k_piezo)} must be N= {N}")
            

        # Setup coupling array for the the microphonics coefficient.
        # Compared to the other two couplings, this value has never been measured since it's difficult 
        # to replicate the conditions. Most of the values are then made up but somewhat realisitic. 
        if k_micro is None:
            # Default: adjacent coupling with k=0
            self.k_micro = np.zeros(N, dtype=np.float32)
        elif isinstance(k_micro, (int, float)):
                self.k_micro = np.full(N, float(k_micro), dtype=np.float32)
        else:
            # Full coupling matrix
            self.k_micro = np.array(k_micro, dtype=np.float32)
            if len(self.k_micro) != N:
                raise ValueError(f"k_piezo array length {len(self.k_micro)} must be N= {N}")
        
        # Setup the time constant of the modes, tau_mode.
        if tau_mode is None:
            # Default: adjacent coupling with k=0
            self.tau_mode = np.ones(N, dtype=np.float32)
        elif isinstance(tau_mode, (int, float)):
                self.tau_mode = np.full(N, float(tau_mode), dtype=np.float32)
        else:
            # Full coupling matrix
            self.tau_mode = np.array(tau_mode, dtype=np.float32)
            if len(self.tau_mode) != N:
                raise ValueError(f"tau_mode array length {len(self.tau_mode)} must be N= {N}")



        # Gymnasium spaces  
        # Action space: piezo voltage only
        self.action_space = spaces.Box(
            low=-max_piezo, high=max_piezo, shape=(), dtype=np.float32
        )

        # Observation space: position and velocity of the observation oscillator
        obs_dim = 2  # oscillator position + velocity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state arrays
        self.cavity_voltage = None
        self.detuning_mode = None
        self.detuning_total = None
        self.dotdetuning_mode = None
        self.time = None
        self.step_count = None
        
        # Data storage for analysis
        self.time_history = []
        self.cavity_voltage_history = []
        self.detuning_total_history = []
        self.drive_history = []
        
        # Reset to initialize
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the simulation to initial conditions (Gymnasium interface)."""
        super().reset(seed=seed)
<<<<<<< HEAD
        if vforward is None:
            self.cavity_voltage = np.zeros(1, dtype=np.complex128)
            self.y = np.concatenate([np.zeros(2 * self.N), [0.0, 0.0]])
        else: 
            self.cavity_voltage = 2*vforward
            self.y = np.concatenate([np.zeros(2 * self.N), [2*vforward, 0.0]])

=======
        
        self.cavity_voltage = np.zeros(1, dtype=np.complex128)
>>>>>>> parent of 79a8679 (Reapply "Added ANC and change dt")
        self.detuning_mode = np.zeros(self.N, dtype=np.float32)
        self.dotdetuning_mode = np.zeros(self.N, dtype=np.float32)
        self.detuning_total = np.zeros(1,dtype=np.float32)
        self.time = 0.0
        self.step_count = 0

        self.time_history = []
        self.cavity_voltage_history = []
        self.detuning_total_history = []
        self.drive_history = []

        # Add small random perturbations for varied initial conditions
        if seed is not None:
            np.random.seed(seed)
        self.detuning_mode += np.random.normal(0, 0.01, self.N).astype(np.float32)
        self.dotdetuning_mode += np.random.normal(0, 0.01, self.N).astype(np.float32)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for Gymnasium interface."""
        # Return the cavity detuning for the modes 
        obs = np.array([
            self.detuning_mode,  # Observation cavity detuning mode
            self.dotdetuning_mode  # Observation of derivative of the cavity detuning mode
        ], dtype=np.float32)
        
        # Add Gaussian noise to observations if noise standard deviation > 0
        if self.observation_noise_std > 0:
            noise = np.random.normal(0, self.observation_noise_std, size=obs.shape).astype(np.float32)
            obs += noise
        
        return obs


    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for Gymnasium interface."""
        
        return {
            'time': self.time,
            'step_count': self.step_count,
            'drive_force': self.drive_history[-1] if self.drive_history else 0.0
        } 
   
    def cavity_dynamics(self,t,y,drive):
        """
        Defines the system of first-order voltage calculation, solving for V.
        It also 
        The state vector y is [Dw1...DwN, dDwdt1...dDwdtN, v_real, v_imag].
        """

        #This will hold the input power, needs be implemented and the piezo [0]  and [1] to the piezo. 
        # Handle action input

        """        t_mod = t % self.repetition_period  # Modulo with repetition period

        # Forward voltage calculation (from original cavity_test.py)
        if self.RL is not None and self.Amp is not None:
            forward_voltage = self.RL * self.Amp * ( 
                (t_mod >= self.t1) * (t_mod < self.tfill) + 
                self.ratio * (t_mod >= self.tfill) * (t_mod < self.tflat) 
            )
        else:
            forward_voltage = 0.0  # No forward power if parameters not set
        """    
        forward_voltage = drive[0]    
        #********** Microphonics calculation********************************************
        # The slow variation models the drift in pressure of the helium bath of the cavity  
        microphonics_slow = 0.01 * np.sin(2 * np.pi * 0.5 * t)*0 
        # the sinousoid model vacuum pumps, other static vibrations. 
        microphonics_sinousoid = np.sum( 0.05 * np.sin(self.wfreq_comp * t))
        microphonics = microphonics_slow + microphonics_sinousoid
    
        Dw_modes = y[0:self.N]  #detuning modes, /Delta /omega
        dDwdt_modes = y[self.N:2*self.N]  #derivative of detuning modes, d(/Delta /omega)/dt
        vcavity_real = y[2*self.N] #cavity voltage real
        vcavity_imag = y[2*self.N + 1] #cavity voltage imaginary         
        vcavity_complex = vcavity_real + 1j * vcavity_imag

        detuning_total = np.sum(Dw_modes)
        piezo = np.clip(drive[1], -self.max_piezo, self.max_piezo)   


        # Derivatve of the cavity voltage 
        dvdt_complex = (-self.w_half + detuning_total * 1j) * vcavity_complex + 2 * self.w_half * forward_voltage

        k_LFD = self.k_LFD
        k_piezo = self.k_piezo
        k_micro = self.k_micro
        Leff=self.Leff
        Angw_squared = self.angular_mech_w**2
        # Second derivate of the detuning modes
        D2wdt2_modes = - (2 / self.tau_mode) * dDwdt_modes - Angw_squared * Dw_modes -Angw_squared * k_LFD * np.abs((vcavity_complex-2*forward_voltage) /(1e6*Leff))**2 + Angw_squared * k_piezo * piezo + Angw_squared * k_micro * microphonics    
        dv_real_dt = dvdt_complex.real
        dv_imag_dt = dvdt_complex.imag

        # Combine all derivatives into a single flat array to return
        return np.concatenate([dDwdt_modes, D2wdt2_modes, [dv_real_dt, dv_imag_dt]])

    #The step def does the calculation of the cavity voltage and the detuning
    def step(self, action):
        """
        Advance the simulation by one time step (Gymnasium interface).
        
        Parameters:
        -----------
        action : np.ndarray or float
            Drive force applied to oscillator 0
        """
        forward_voltage = float(action[0]) # units of V 
        # --- RK4 Method ---
        k1 = self.dt * self.cavity_dynamics(self.time, self.y,action)
        k2 = self.dt * self.cavity_dynamics(self.time + 0.5 * self.dt, self.y + 0.5 * k1 * self.dt, action)
        k3 = self.dt * self.cavity_dynamics(self.time + 0.5 * self.dt, self.y + 0.5 * k2 * self.dt, action)
        k4 = self.dt * self.cavity_dynamics(self.time + self.dt, self.y + k3*self.dt, action)
            
        # Update state and time, note that y is [Dw1...DwN, dDwdt1...dDwdtN, v_real, v_imag].
        self.y += (k1 + 2*k2 + 2*k3 + k4) / 6.0 # the result is k+1 , step of dt
        detuning_mode = self.y[0:self.N]
        vcavity_real = self.y[2*self.N]
        vcavity_imag = self.y[2*self.N + 1]
        self.cavity_voltage = vcavity_real + vcavity_imag*1j

<<<<<<< HEAD
        self.detuning_total=np.sum(detuning_mode)
=======
        # Uses simple Euler method to solve equations  
        # Update values for the detuning modes, and all modes 
        #Solving for cavity voltage, 1st order ODE, Note that the drive force has the term RL 
        V_k = self.cavity_voltage # V[k]
        right_term_cavity = (-self.w_half + self.detuning_total * 1j) * self.cavity_voltage + 2 * self.w_half * forward_voltage
        self.cavity_voltage += right_term_cavity* self.dt #This will end up being v[k+1]
       
        k_LFD = self.k_LFD
        k_piezo = self.k_piezo
        k_micro = self.k_micro
        Ang_w = self.angular_mech_w**2
        Leff=self.Leff
        
        DW_k=self.detuning_mode # will be in place of DW [k]
        self.detuning_mode += self.dotdetuning_mode * self.dt # this is DW [k+1]
        right_term_mode = -k_LFD* Ang_w * np.abs(V_k /(1e6*Leff) )**2 + k_piezo* Ang_w * piezo + k_micro* Ang_w * microphonics -(2 / self.tau_mode) * self.dotdetuning_mode - Ang_w * DW_k
        self.dotdetuning_mode += right_term_mode * self.dt

        self.detuning_total=np.sum(self.detuning_mode)

        #print(f"V_k {np.abs(V_k / 1e6)**2}")
>>>>>>> parent of 79a8679 (Reapply "Added ANC and change dt")

        # Update time and step count
        self.time += self.dt
        self.step_count += 1
        
        # Store data for analysis
        self.time_history.append(self.time)
        self.detuning_total_history.append(self.detuning_total)
        self.cavity_voltage_history.append(self.cavity_voltage.copy())
        self.drive_history.append(forward_voltage)  # Now storing force instead of position
        
        # Check if episode is done
        terminated = False  # This environment doesn't have a natural termination
        truncated = self.step_count >= self.max_episode_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, terminated, truncated, info
    
    # Need to plot plot the total detuning and the voltage of the cavity, not sure of the fft. 
    def plot_results(self):
        """
        Plot simulation results.The drive for the cavity, cavity voltage, and total detuning. 
        
        Parameters:
        -----------
        show_all : bool
            If True, plot all oscillators. If False, plot only drive and last.
        """
        if not self.time_history:
            print("No data to plot. Run simulation first.")
            return
        
        times = np.array(self.time_history)
        detuning = np.array(self.detuning_total_history)
        cavity_voltage = np.array(self.cavity_voltage_history)
        drives = np.array(self.drive_history)
        
        plt.figure(figsize=(12, 8))
        
        # Plot drive signal
        plt.subplot(2, 1, 1)
        plt.plot(times, drives/(1e6), 'r-', alpha=0.5,linewidth=2, label='Drive Signal')
        plt.plot(times, np.abs(cavity_voltage)/1e6, 'r--', alpha=0.7, label='Cavity Voltage')
        plt.xlabel('Time [s]')
        plt.ylabel('Drive and Cavity Voltage [MV]')
        plt.title('Drive and Cavity Response')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        #Not sure we need this 
        # Plot phase relationship
        plt.subplot(2, 1, 2)
        plt.plot(times, detuning/ (2*np.pi), 'g-', alpha=0.7)
        plt.xlabel('Time [s]')
        plt.ylabel('Cavity Detuning [Hz]')
        plt.title('Cavity Detuning')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def render(self, mode='human'):
        """Render the environment (Gymnasium interface)."""
        if mode == 'human':
            # Simple text-based rendering
            if self.positions is not None:
                print(f"Time: {self.time:.3f}, Step: {self.step_count}, "
                      f"Drive Force: {self.drive_history[-1] if self.drive_history else 0:.3f}, "
                      f"Total Detuning: {self.detuning_total[-1]:.3f}")
        elif mode == 'rgb_array':
            # Could implement matplotlib-based rendering here
            # For now, return None
            return None
    
    def close(self):
        """Close the environment (Gymnasium interface)."""
        # Clean up any resources if needed
        plt.close('all')
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system configuration."""
        return {
            'N': self.N,
            'k_LFD': self.k_LFD.copy(),
            'w_half': self.w_half,
            'dt': self.dt,
            'max_force': self.max_force,
            'observation_noise_std': self.observation_noise_std,
        }    
    
    def set_observation_noise(self, noise_std: float):
        """Set the standard deviation of observation noise."""
        if noise_std < 0:
            raise ValueError("Noise standard deviation must be non-negative")
        self.observation_noise_std = noise_std
    
if __name__ == "__main__":

    """
    For comprehensive examples and validation tests, run:
    python validation.py
    
    For PID controller usage, import from pid_controller:
    from pid_controller import PIDController
    """
    print("CoupledOscillators simulation system")
    print("For examples and validation tests, run: python validation.py")
    print("For PID controller, import: from pid_controller import PIDController")
