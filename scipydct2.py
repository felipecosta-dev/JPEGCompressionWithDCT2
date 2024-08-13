from scipy.fft import dct, idct

class SciPyDCT2:
    @staticmethod
    def dct2(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def idct2(block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
