import numpy as np
from typing import Optional, Tuple, Dict, Union
from pathlib import Path


class ANC_LMS_Control:
    """
    ANC (Active Noise Control) with LMS (Least Mean Square) controller resonance control     
    
    Computes the piezo voltage needed to compensate for mechanical vibrations. The frequencies that need to compenesated 
    are known and are obtaind from an FFT. 

    """
    
    def __init__(self, mu: float = 0.001, eta: float = 0.001, wfreq_comp = None, dt: float = 0.01, 
                 output_limits: Optional[Tuple[float, float]] = None,
                 config_file: Optional[Union[str, Path]] = None):
        """
        Initialize PID controller.
        
        Parameters:
        -----------
        mu : float
            learning rate for the amplitudes of the piezo drive, this value should be less than 1.
        eta : float
            learning rate for the phase delay of the tuner, this value should be less than 1.     
        wfreq_comp : float
            frequencies to be compensated. These are known based on the FFT. 
        dt : float
            Time step (should match simulation dt)
        output_limits : tuple, optional
            (min, max) limits for controller output force
        integral_limits : tuple, optional
            (min, max) limits for integral term to prevent windup
        config_file : str, Path, or None
            Path to configuration file (JSON or YAML). If provided, overrides other parameters.
        """
        # Load configuration from file if provided
        if config_file is not None:
            try:
                from .config import ExperimentConfig
            except ImportError:
                from config import ExperimentConfig
            
            config = ExperimentConfig(config_file)
            ANC_config = config.get_ANC_config()
            
            # Override parameters with values from config file
            mu = ANC_config.get('mu', mu)
            eta = ANC_config.get('eta', eta)
            wfreq_comp = ANC_config.get('wfreq_comp', wfreq_comp)
            output_limits = ANC_config.get('output_limits', output_limits)
            
            # Get dt from simulation config if not explicitly set in PID config
            sim_config = config.get_simulation_config()
            dt = sim_config.get('dt', dt)
        
        self.mu = mu
        self.eta = eta
        self.wfreq_comp = wfreq_comp
        self.dt = dt
        self.output_limits = output_limits
                
        # Internal state
        self.reset()
    
    def reset(self):
        """Reset controller internal state."""
        length= len(self.wfreq_comp)
        self.I_mk= np.zeros(length)
        self.Q_mk = np.zeros(length)
        self.phi_mk = np.zeros(length)
        self.last_time = None
    
    def update(self, detuning: float, t: float) -> float:
        """
        Compute ANC control output.
        
        Parameters:
        -----------
        detuning : float
            detuning of the cavity
            
        Returns:
        --------
        float
            Control voltage to apply to the piezo
        """
        #Currently I_m is ill defined, all other terms also 
        I_m = self.I_mk
        Q_m = self.Q_mk
        phi_m = self.phi_mk 

        #calculate the coefficients fo the amplitude and phase 
        #algorithm based on https://arxiv.org/abs/2209.13896
        I_m += -self.mu*detuning*np.cos(self.wfreq_comp*t-phi_m) #calculates the k+1
        Q_m += -self.mu*detuning*np.sin(self.wfreq_comp*t-phi_m) #calculates the k+1   
        phi_m += -self.eta*detuning*(self.I_mk*np.sin(self.wfreq_comp*t-phi_m)-self.Q_mk*np.cos(self.wfreq_comp*t-phi_m)) #calculates the k+1

        #Individual piezo driving terms
        piezo_m=self.I_mk *np.cos(self.wfreq_comp*t) + self.Q_mk*np.sin(self.wfreq_comp*t)

        # Calculate total output, this is based on the input from the user
        #output = proportional + integral + derivative
        output = np.sum(piezo_m)


        # Apply output limits
        if self.output_limits:
            output = np.clip(output, *self.output_limits)
        
        # Store for next iteration
        self.I_mk = I_m
        self.Q_mk = Q_m
        self.phi_mk = phi_m
        
        return output
    
    def set_limits(self, output_limits: Optional[Tuple[float, float]] = None):
        """Update controller limits."""
        self.output_limits = output_limits