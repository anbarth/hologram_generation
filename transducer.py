import numpy as np

class Transducer:
    def __init__(self, pixels, phase=0, amp=0):
        self.pixels = pixels
        self.phase = phase % (2*np.pi)
        self.amp = amp

    def set_phase(self,phase):
        self.phase = phase % (2*np.pi)

    def set_amp(self,amp):
        self.amp = amp
    