import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage
import math


def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
   centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
   centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)
 
   def gaussian(i,j):
      coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
      return 1 - coefficient if highPass else coefficient
 
   return numpy.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])
 
def filterDFT(imageMatrix, filterMatrix):
   shiftedDFT = fftshift(fft2(imageMatrix))
   filteredDFT = shiftedDFT * filterMatrix
   return ifft2(ifftshift(filteredDFT))
 
def lowPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))
 
def highPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))
   

def hybridImage(highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
   highPassed = highPass(highFreqImg, sigmaHigh)
   lowPassed = lowPass(lowFreqImg, sigmaLow)
 
   return highPassed + lowPassed
   
if __name__ == "__main__":
   marilyn = ndimage.imread("marilyn.png", flatten=True)
   lowPassedMarilyn = lowPass(marilyn, 20)
   misc.imsave("low-passed-marilyn.png", numpy.real(lowPassedMarilyn))
   einstein = ndimage.imread("einstein.png", flatten=True)
   highPassedeinstein = highPass(einstein, 20)
   misc.imsave("high-passed-einstein.png", numpy.real(highPassedeinstein))
   
   HI = hybridImage(highPassedeinstein, lowPassedMarilyn, 20, 20)
   misc.imsave("HybridImage.png", numpy.real(HI))
   




