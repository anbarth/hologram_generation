import numpy as np
from importlib import reload
import transducer
reload(transducer)

# a: size length in px of each transducer
class TransducerArray:
    def __init__(self,N_grid,N_array,a,x,y):
        self.N_array = N_array
        self.N_grid = N_grid
        
        self.N_transducers = N_array*N_array
        self.phase = [0]*self.N_transducers
        self.amp = [1]*self.N_transducers

        # now make N_array^2 transducers in appropriate locations
        self.transducers = []
        transducer_id=0
        for i in range(N_array):
            for j in range(N_array):
                upper_left_x = x+i*a
                upper_left_y = y+j*a
                pixels = [(p,q) for p in range(upper_left_x,upper_left_x+a) for q in range(upper_left_y,upper_left_y+a)]
                trans = transducer.Transducer(pixels,phase=0,amp=1)
                self.transducers.append(trans)
                transducer_id += 1
        
                
    def set_amp_list(self,amp):
        if not len(amp)==self.N_transducers:
            print('Bad size for phase array')
            return
        for i in range(self.N_transducers):
            self.set_amp(amp[i],i)
    
    def set_phase_list(self,phase):
        if not len(phase)==self.N_transducers:
            print('Bad size for phase array')
            return
        for i in range(self.N_transducers):
            self.set_phase(phase[i],i)
    
    def get_amp_list(self):
        return self.amp

    def get_phase_list(self):
        return self.phase


    def set_amp(self,amp,transducer_id):
        self.transducers[transducer_id].set_amp(amp)
        self.amp[transducer_id]=amp
    
    def set_phase(self,phase,transducer_id):
        self.transducers[transducer_id].set_phase(phase)
        self.phase[transducer_id]=phase
    
    def complex_pressure_grid(self):
        p = np.zeros((self.N_grid,self.N_grid))
        for transducer_id in range(self.N_transducers):
            trans = self.transducers[transducer_id]
            # get amp and phase, then loop over pixels and set them
            for px in trans.pixels:
                p[px[0],px[1]] = trans.amp*np.exp(1j * trans.phase)
        return p
    
    def pressure_grid(self):
        p = np.zeros((self.N_grid,self.N_grid))
        for transducer_id in range(self.N_transducers):
            trans = self.transducers[transducer_id]
            for px in trans.pixels:
                p[px[0],px[1]] = trans.amp
        return p