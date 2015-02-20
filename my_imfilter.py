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
   image1 = ndimage.imread("bicycle.png", flatten=True)
   lowPassed_image1 = lowPass(image1, 20)
   misc.imsave("low-passed-bicycle.png", numpy.real(lowPassed_image1))
   image2 = ndimage.imread("motorcycle.png", flatten=True)
   highPassed_image2 = highPass(image2, 20)
   misc.imsave("high-passed-motorcycle.png", numpy.real(highPassed_image2))
   
   HI = hybridImage(highPassed_image2, lowPassed_image1, 20, 20)
   misc.imsave("HybridImage1.png", numpy.real(HI))
   

   
if __name__ == "__main__":
   image1 = ndimage.imread("marilyn.png", flatten=True)
   lowPassed_image1 = lowPass(image1, 20)
   misc.imsave("low-passed-marilyn.png", numpy.real(lowPassed_image1))
   image2 = ndimage.imread("einstein.png", flatten=True)
   highPassed_image2 = highPass(image2, 20)
   misc.imsave("high-passed-einstein.png", numpy.real(highPassed_image2))
   
   HI = hybridImage(highPassed_image2, lowPassed_image1, 20, 20)
   misc.imsave("HybridImage2.png", numpy.real(HI))
   

if __name__ == "__main__":
   image1 = ndimage.imread("fish.png", flatten=True)
   lowPassed_image1 = lowPass(image1, 20)
   misc.imsave("low-passed-fish.png", numpy.real(lowPassed_image1))
   image2 = ndimage.imread("submarine.png", flatten=True)
   highPassed_image2 = highPass(image2, 20)
   misc.imsave("high-passed-submarine.png", numpy.real(highPassed_image2))
   
   HI = hybridImage(highPassed_image2, lowPassed_image1, 20, 20)
   misc.imsave("HybridImage3.png", numpy.real(HI))
   

if __name__ == "__main__":
   image1 = ndimage.imread("plane.png", flatten=True)
   lowPassed_image1 = lowPass(image1, 20)
   misc.imsave("low-passed-plane.png", numpy.real(lowPassed_image1))
   image2 = ndimage.imread("bird.png", flatten=True)
   highPassed_image2 = highPass(image2, 20)
   misc.imsave("high-passed-bird.png", numpy.real(highPassed_image2))
   
   HI = hybridImage(highPassed_image2, lowPassed_image1, 20, 20)
   misc.imsave("HybridImage4.png", numpy.real(HI))
   

if __name__ == "__main__":
   image1 = ndimage.imread("cat.png", flatten=True)
   lowPassed_image1 = lowPass(image1, 20)
   misc.imsave("low-passed-cat.png", numpy.real(lowPassed_image1))
   image2 = ndimage.imread("dog.png", flatten=True)
   highPassed_image2 = highPass(image2, 20)
   misc.imsave("high-passed-dog.png", numpy.real(highPassed_image2))
   
   HI = hybridImage(highPassed_image2, lowPassed_image1, 20, 20)
   misc.imsave("HybridImage5.png", numpy.real(HI))
   

