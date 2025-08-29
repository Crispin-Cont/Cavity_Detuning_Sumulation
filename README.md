# Cavity_Detuning_Simulation
Cavity simulation with Gymnasium interface

## 🚀 Quick Start Guide

### Prerequisites
```bash
pip install numpy matplotlib gymnasium
```

### 1. Run Basic Examples
```bash
python cavity_test.py
```
The Cavity_Simulation_Euler.ipynb also a walk through. 

# Brief Introduction to Cavity Model

The cavity main cavity mode ($\pi$) will be modeled coupled to the mechanical resonance of the system. This is given by the equations: 

This describes the voltage of the cavity. It's a second-order equation, but we're only concerned with slow oscillations, specifically those in the range of 1 to 500 Hz. The driving terms ($ I_F$) of the cavities of interest are 325 MHz, 650 MHz, and 1.3 GHz. 

The equation is given by many []:

$
 \dot{V} + (\omega_{1/2}-i \Delta \omega)V = 2\omega_{1/2} R_L I_F
$

The equation for the mechanical modes is given by [] : 

$
    \ddot{\Delta \omega_m } + 2/\tau_m \dot{\Delta \omega_m} + \Omega_m^2 \Delta \omega_m = -k_m^{LFD} \Omega_m^2 E_{acc}^2 + k_m^{piezo} \Omega_m^2 P + k_m^{micro} \Omega_m^2 M
$

Where $k_m^{LFD}$ is the Lorentz force detuning (LFD) coefficient in units of $ Hz/(MV/m)^2$, $k_m^{piezo}$ is the piezo coupling to the cavity in units of $ Hz/V $, and $k_m^{micro}$ is the coupling of microphonics to the cavity modes in units of $Hz/Torr$. Note that the LFD and piezo couplings can be measured, while the microphonics have not been measured. The accelerating gradient of the cavity is given by $E_{acc} = V/L{eff}$, the piezo driving term is P, and the microphonics term is M. The total detuning is then 

$
    \Delta \omega =\sum_m  \Delta \omega_m 
$

The goal then is to decrease  $\Delta \omega$ to a certain RMS (say <1 Hz) by actuating the piezo (P). The piezo voltage is limited to -120 V to 120 V. 

# Solving the ODE
The coupled equations are solved by the Euler integration method. This is done in the cavity_simulator.py file in the step() function. The largest step possible in the simulation is $dt = 10^{-6} s$; a larger value will produce numerical instability. RK4 integration will be implemented later to see if there is an improvement in the step size. 