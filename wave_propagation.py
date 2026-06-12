import tensorflow as tf
import numpy as np
import math
import output_display

# creates propagator H
# k: wavenumber=2pi/lambda=2pi f/c in propagation medium (1/m)
# N: square grid size length (px)
# z: propagation distance (m)
# dx: pixel size (m) 
def band_limited_propagator(k, N, z, dx):
    L = N*dx

    # k_vec = -Npi/L ... (N-1)pi/L
    k_vec = np.arange(-N, N)
    k_vec = k_vec*math.pi / L
    
    # shift so that k_vec = 0 ... (N-1)pi/L, -Npi/L... -pi/L
    k_vec = np.fft.ifftshift(k_vec)

    # create propagator H(kx, ky)=exp(i z sqrt(k^2-kx^2-ky^2))
    kx, ky = np.meshgrid(k_vec, k_vec)
    kz = pow(k**2 -  (pow(kx, 2) + pow(ky, 2))+0j, 0.5)
    #H = np.conj(np.exp(1j * z * kz))
    H = np.exp(1j * z * kz)
    
    # cutoff k
    D = (2*N-1)*dx
    k_cutoff = k * pow(0.5 * D**2 / (0.5 * D**2 + z**2) , 0.5)
    #k_cutoff = k* 2*L / pow(L**2 + z**2 , 0.5)

    # zero out any k vectors that exceed the cutoff
    k_transverse = pow(pow(kx,2) + pow(ky, 2), 0.5)
    H = np.where(k_transverse > k_cutoff, 0, H)
    return H


# creates propagator H
# k: wavenumber=2pi/lambda=2pi f/c in propagation medium (1/m)
# N: square grid size length (px)
# z1: propagation distance (m)
# dx: pixel size (m) 
# z2: distance to top plate (m)
# R1: reflection coefficient from medium to hologram
# R2: reflection coefficient from medium to top plate
def band_limited_propagator_with_reflections_up(k, N, z1, dx, z2, R1, R2):
    L = N*dx

    # k_vec = -Npi/L ... (N-1)pi/L
    k_vec = np.arange(-N, N)
    k_vec = k_vec*math.pi / L
    
    # shift so that k_vec = 0 ... (N-1)pi/L, -Npi/L... -pi/L
    k_vec = np.fft.ifftshift(k_vec)

    # create propagator H(kx, ky)=exp(i z sqrt(k^2-kx^2-ky^2))
    kx, ky = np.meshgrid(k_vec, k_vec)
    kz = pow(k**2 -  (pow(kx, 2) + pow(ky, 2))+0j, 0.5)
    factor1 = np.exp(1j * z1 * kz)
    factor2 = 1 + R2*np.exp(1j * 2*z2 * kz)
    factor3 = 1 / (1-R1*R2*np.exp(1j * 2*(z1+z2) * kz))
    H = factor1*factor2*factor3
    
    # cutoff k
    D = (2*N-1)*dx
    k_cutoff = k * pow(0.5 * D**2 / (0.5 * D**2 + z1**2) , 0.5)
    #k_cutoff = k* 2*L / pow(L**2 + z**2 , 0.5)

    # zero out any k vectors that exceed the cutoff
    k_transverse = pow(pow(kx,2) + pow(ky, 2), 0.5)
    H = np.where(k_transverse > k_cutoff, 0, H)
    return H


def band_limited_propagator_with_reflections_down(k, N, z1, dx, z2, R1, R2):
    L = N*dx

    # k_vec = -Npi/L ... (N-1)pi/L
    k_vec = np.arange(-N, N)
    k_vec = k_vec*math.pi / L
    
    # shift so that k_vec = 0 ... (N-1)pi/L, -Npi/L... -pi/L
    k_vec = np.fft.ifftshift(k_vec)

    # create propagator H(kx, ky)=exp(i z sqrt(k^2-kx^2-ky^2))
    kx, ky = np.meshgrid(k_vec, k_vec)
    kz = pow(k**2 -  (pow(kx, 2) + pow(ky, 2))+0j, 0.5)
    factor1 = np.exp(-1j * z1 * kz)
    factor2 = 1 / (1-R1*R2*np.exp(-1j * 2*(z1+z2) * kz))
    H = factor1*factor2*factor2
    
    # cutoff k
    D = (2*N-1)*dx
    k_cutoff = k * pow(0.5 * D**2 / (0.5 * D**2 + z1**2) , 0.5)
    #k_cutoff = k* 2*L / pow(L**2 + z**2 , 0.5)

    # zero out any k vectors that exceed the cutoff
    k_transverse = pow(pow(kx,2) + pow(ky, 2), 0.5)
    H = np.where(k_transverse > k_cutoff, 0, H)
    return H

# p_in: input complex-valued pressure field
def propagate_broken(p_in, H):
    #N = p_in.shape[0]

    p_in_hat = tf.signal.fft2d(p_in)
    p_out= tf.signal.ifft2d(tf.math.multiply(p_in_hat, H))

    return p_out


def propagate(p_in, H):
    N = p_in.shape[0]
    p_in = tf.constant(p_in,dtype=tf.complex128)
    # pad out the pressure matrix with 0s... for reasons.......
    p_in_padded = tf.pad(p_in, ((N//2,N//2), (N//2,N//2)), 'constant')
    N_padded = p_in_padded.shape[0]

    # fft, apply propagator, and ifft back
    p_in_hat = tf.signal.fft2d(p_in_padded)
    p_out_padded = tf.signal.ifft2d(tf.math.multiply(p_in_hat, H))

    # remove padding around matrix
    N_padded = p_out_padded.shape[0]
    N = N_padded // 2
    start_num = N//2
    end_num = N_padded - N//2
    p_out = p_out_padded[start_num:end_num, start_num:end_num]

    return p_out

def transmission_coefficient(Zt,Zh,Zm,kh,T):
    denom1 = Zh*(Zt+Zm)*np.cos(kh*T)
    denom2 = (Zh*Zh+Zt*Zm)*np.sin(kh*T)
    return 4*Zt*Zh*Zh*Zm / (denom1*denom1 + denom2*denom2)

def loss_func_iasa(target_pressure,current_pressure):

    current_intensity = tf.abs(current_pressure*tf.math.conj(current_pressure))
    target_intensity = tf.abs(target_pressure*tf.math.conj(target_pressure))

    loss = tf.reduce_sum(pow(target_intensity-current_intensity,2))

    return loss



# p_0_amp: pressure amplitude (Pa) at z=0 (transducer). should be a tensor
def iasa(p_0_amp,p_target_amp,H_up,H_down=None,drawPhaseMap=False,drawPressureImage=False,H=0,kh=0,Zt=0,Zh=0,Zm=0):

    # initialize phase at z=0
    N = p_0_amp.shape[0]
    phase_0 = np.zeros((N,N))

    # set up propagators
    #H_up = band_limited_propagator(k, N, z, dx)
    if H_down is None:
        H_down = tf.math.conj(H_up)

    for step_num in range(600):
        # propagate to image plane
        p_0 = p_0_amp*np.exp(1j*phase_0)

        # get alpha_T
        alpha_T = np.ones(p_0.shape)
        if H != 0:
            delta_z = (phase_0 % (2*np.pi)) / (2*np.pi) * H
            T = H-delta_z
            alpha_T = transmission_coefficient(Zt,Zh,Zm,kh,T)

        p_z = np.sqrt(alpha_T)*propagate(p_0, H_up)

        # record loss function
        #loss_record.append(loss_func_iasa(p_z,p_target_amp))
        
        # generate images
        if ((step_num)%100)==0:
            if drawPhaseMap:
                output_display.draw_phase(phase_0)
            if drawPressureImage:
                output_display.draw_pressure_complex(p_z)

        # in image plane, keep new phase but fix amplitude to target 
        phase_z = tf.math.angle(p_z).numpy()
        p_z = p_target_amp*np.exp(1j*phase_z)

        # propagate back
        p_0 = propagate(p_z, H_down)

        # in transducer plane, keep new phase but fix amplitude to input
        phase_0 = tf.math.angle(p_0).numpy()
        # set phase to 0 where transducer does not exist
        phase_0 = np.where(p_0_amp  == 0, 0, phase_0)
        p_0 = p_0_amp*np.exp(1j*phase_0)
            

    return phase_0


def diffPAT(p_0_amp,p_target_amp,H_up,drawPhaseMap=False,drawPressureImage=False,H=0,kh=0,Zt=0,Zh=0,Zm=0):
    
    # initialize phase at z=0
    N = p_0_amp.shape[0]
    phase_0 = np.zeros((N,N))
    phase_0_tf = tf.Variable(phase_0,dtype=tf.float64)
    #phase_0_tf = tf.dtypes.cast(p_0_amp, tf.complex128)

    target_intensity = tf.abs(p_target_amp*tf.math.conj(p_target_amp))

    def loss_func_diffPAT(phase_0_tf):
        # create pressure field at z=0 and propagate to image plane
        phase_exp = tf.dtypes.complex(tf.math.cos(phase_0_tf), tf.math.sin(phase_0_tf))
        p_0 = tf.math.multiply(p_0_amp, phase_exp)

        # get alpha_T
        alpha_T = np.ones(p_0.shape)
        if H != 0:
            delta_z = (phase_0 % (2*np.pi)) / (2*np.pi) * H
            T = H-delta_z
            alpha_T = transmission_coefficient(Zt,Zh,Zm,kh,T)

        p_z = np.sqrt(alpha_T)*propagate(p_0, H_up)

        # compare image plane intensity with target
        current_intensity = tf.abs(p_z * tf.math.conj(p_z))
        # if you just add a term here like phase^2, it should reward staying near 0
        #loss = tf.reduce_sum(tf.math.abs(current_intensity-target_intensity)+0.01*tf.math.multiply(phase_0_tf,phase_0_tf))
        loss = tf.reduce_sum(tf.math.abs(current_intensity-target_intensity))
        return loss

    # Get optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)

    for step_num in range(200): #200

        with tf.GradientTape() as tape:
            loss = loss_func_diffPAT(phase_0_tf)

        # Compute the gradient of the loss with respect to x
        grads = tape.gradient(loss, [phase_0_tf])

        # Apply the gradient using the optimizer
        opt.apply_gradients(zip(grads, [phase_0_tf]))

        # Evaluate performance
        #loss_record.append(loss_func(p_z,p_target_amp))
    
    return phase_0_tf.numpy()


def diffPAT_array(transducer_array,p_target_amp,H_up,drawPhaseMap=False,drawPressureImage=False):
    
    # initialize phase at z=0
    #p_array_amp = transducer_array.get_amp_list()
    phase_pat = transducer_array.get_phase_list()
    phase_pat_tf = tf.Variable(phase_pat,dtype=tf.float64)
    print(phase_pat_tf)

    target_intensity = tf.abs(p_target_amp*tf.math.conj(p_target_amp))

    
    def loss_func_diffPAT_array(phase_pat_tf):
        # create pressure field at z=0 and propagate to image plane
        # this needs to be modified to call transducer_array.complex_pressure_grid()
        transducer_array.set_phase_list(phase_pat_tf.numpy())
        p_0 = transducer_array.complex_pressure_grid()
        p_z = propagate(p_0, H_up)

        # compare image plane intensity with target
        current_intensity = tf.abs(p_z * tf.math.conj(p_z))
        loss = tf.reduce_sum(tf.math.abs(current_intensity-target_intensity))
        return loss

    # Get optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)

    for step_num in range(1):

        with tf.GradientTape() as tape:
            loss = loss_func_diffPAT_array(phase_pat_tf)

        # Compute the gradient of the loss with respect to x
        grads = tape.gradient(loss, [phase_pat_tf])

        # Apply the gradient using the optimizer
        opt.apply_gradients(zip(grads, [phase_pat_tf]))

        # Evaluate performance
        #loss_record.append(loss_func(p_z,p_target_amp))
    
    return phase_pat_tf.numpy()



def calculate_alpha(phase, f0):
    # Hologram
    c_h = 2246
    rho_h = 1240
    Z_h = c_h * rho_h
    k_h = (2*math.pi*f0) / c_h
    T0=1
    
    #transducer surface (assume gel to be impedance matched with hologram)
    Z_t = Z_h
    # water
    rho_m = 1000
    c_m = 1480
    Z_m = c_m * rho_m
    k_m = (2*math.pi*f0) / c_m
    #Hologram Thickness
    delta_phase = phase % (2*math.pi)
    delta_T = delta_phase / (k_m-k_h)
    Thickness = T0 - delta_T

    alpha_t_top = (4*Z_t*(Z_h**2)*Z_m)
    alpha_t_bot_1 = (Z_h**2 * (Z_t + Z_m)**2) * pow(math.cos((k_h * Thickness)), 2)
    alpha_t_bot_2 = (Z_h**2 + Z_t*Z_m)**2 * pow(math.sin((k_h * Thickness)), 2)

    alpha_t = pow(alpha_t_top / (alpha_t_bot_1 + alpha_t_bot_2), 0.5)
    return alpha_t