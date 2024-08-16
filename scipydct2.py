from scipy.fft import dct, idct

class SciPyDCT2:
    def dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
        
    def dct1d(self, block):
        return dct(block, norm='ortho')
    
    def idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
