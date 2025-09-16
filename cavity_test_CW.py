import numpy as np
from Cavity_Simulator import Cavity_MechanicalModes
from ANC_LMS_controller import ANC_LMS_Control
#from pid_controller import PIDController
import matplotlib.pyplot as plt
import time
# Example usage and demonstration
if __name__ == "__main__":
    start_time=time.time() # record the start time
    
    # Cavity parameters (650 MHz, low beta)
    #Coupling parameter of the cavity 
    Qext = 3e6
    Q0 = 2.7e10
    beta = Q0 / Qext
    QL = Q0 / (1 + beta)
    #Cavity frequency
    w0 = 2 * np.pi * 650e6  # Units in Hz
    RQ=341 # r/Q cavity
    RL = (0.5 * RQ) * QL  # loaded r/Q of cavity, units in Ohm
    #half bandwidth of the cavity 
    whalf = w0 / (2 * QL)

    #effective length of acceleration 
    Leff = 1.038  # Units in Meters
    #Time constant of the cavity, determines filling time and decay
    tau = 2 * QL / w0  # Units in seconds
    
    # Frequency of the mechanical modes, inside the array in Hz (angular frequency)
    O_m = 2 * np.pi * np.array([157, 182, 189, 215, 292, 331, 380, 412, 462, 471])

    #time constant of the mechanical modes 
    tau_m = 2 * np.array([56.8, 113.2, 70.38, 25, 300, 304.2, 409.46, 305.11, 202.19, 205.86]) * 1e-3 # units of s
    
    # These numbers are not realistic; other data exist for this and will later be added
    # Lorentz force detuning, coupling to the voltage,  
    k_L =2*np.pi* 2.9636 * np.array([0.02, 0.03, 0.055, 0.55, 0.085, 0.029, 0.052, 0.19, 0.075, 0.095])  #Units of Hz/(MV/m)^2
    # Piezo coupling, units Hz/V
    k_P =2*np.pi* 16.9348 * np.array([0.02, 0.03, 0.055, 0.55, 0.085, 0.029, 0.052, 0.19, 0.075, 0.095])
    #Microphonics coupling, units Hz/Torr
    k_M =2*np.pi* 64.0982 * np.array([0.02, 0.03, 0.055, 0.55, 0.085, 0.029, 0.052, 0.19, 0.075, 0.095])

    #parameters used to determine the fill time of the cavity, used for forward. 
    t1 = 1e-3
    tfill = t1 + tau * np.log(2)
    tflat = tfill + t1

    #Ratio to produce flat top where the beam is accelerated. 
    ratio = (1 - np.exp(-(tfill - t1) / tau))

    # amplitude constant, the 25e6 is the accelerating gradient in MV/m. 
    Amp = (25e6 * Leff) / RL  # units A



    # the dt value must be smaller than 1e-6 s, incresing the time step will lead to numerical 
    oscillators = Cavity_MechanicalModes(N = 10, Leff=Leff, k_LFD = k_L, k_piezo=k_P,k_micro=k_M, w_half = whalf, tau_mode = tau_m , angular_mech_w= O_m,
                                         dt=0.05e-6)
    wfreq_comp=2*np.pi*np.array([5, 10 ,20 ,50])
    LMS_contoller = ANC_LMS_Control(mu=1e-9, eta = 1e-9, wfreq_comp = wfreq_comp,dt=0.05e-6)

    print("Cavity driven by Lorentz Force Detuning (LFD) Simulation")
    print("=" * 40)
    print(f"Number of oscillators: {oscillators.N}")
    print(f"Lorentz Force Detuning Coefficients: {oscillators.k_LFD}")
    print(f"Mode time constants: {oscillators.tau_mode}")
    print(f"Angular mechanical frquencies: {oscillators.angular_mech_w}")
    print(f"Time step: {oscillators.dt}")
    print()
    LMS_data = {'time': [], 'force1': [], 'force2': []}

    force0 = RL * Amp  # This forward voltage
    obs, info = oscillators.reset(0.5*force0+0j)
    LMS_contoller.reset()
    detuning = 0.0
    for i in range(20000000):  # 1 s. 
        t = oscillators.time
        force1 = LMS_contoller.update(detuning,t)   #piezo 
        force2 = 0.01 * np.sin(2 * np.pi * 5 * t) + 0.01 * np.sin(2 * np.pi * 10 * t) + 0.01 * np.sin(2 * np.pi * 20 * t) +  0.01 * np.sin(2 * np.pi * 50 * t) #microphonics 
        force = [0.5*force0, force1, force2]
        obs, term ,trunc , info = oscillators.step(force) 
        detuning = info['detuning']
        LMS_data['time'].append(t)
        LMS_data['force1'].append(force1)
        LMS_data['force2'].append(force2)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time : {elapsed_time} s")

    oscillators.plot_results()
    print()

    plt.figure(figsize=(12, 10))
        
    # Position tracking
    plt.plot(LMS_data['time'], LMS_data['force1'], 'r', linewidth=2, label='piezo')
    plt.plot(LMS_data['time'], LMS_data['force2'], 'g', linewidth=2, label='microphonics')
    plt.tight_layout()
    plt.show()
    print(f"Simulation completed: {len(oscillators.time_history)} time steps")
    print(f"Final time: {oscillators.time:.2f} s")
    print()
    