'''These Codes Are Written By Mehdi Touyserkani
    Email Address: Ir_Bestpro@yahoo.com
    Website: Https://www.Ir-Bestpro.com
 '''

from skimage import feature
import numpy as np
from scipy.stats import entropy

class GLCM:

	def __init__(self, P, R):

		#_______Get P (number of Neighbours) And R(Radius) for lbp algorithm_____

		self.P = P 
		self.R = R

	def Pattern(self, image, eps=1e-7):

		#________Create Local Binary Pattern With Uniform Method For Rotation Invariant______________________

		lbp = feature.local_binary_pattern(image, self.P,self.R, method="uniform") # Using Uniform Algorithm

		##_____Return Statistical Features______________

		g = feature.graycomatrix(image.astype(np.uint8), [1], [0, np.pi/2], levels=256)
		contrast = feature.graycoprops(g,'contrast')# contrast
		dissimilarity = feature.graycoprops(g,'dissimilarity') #dissimilarity
		homogeneity = feature.graycoprops(g,'homogeneity') # homogeneity
		ASM = feature.graycoprops(g,'ASM') #angular second moment
		energy = feature.graycoprops(g,'energy') # energy
		correlation = feature.graycoprops(g,'correlation') # Correlation
		ent = entropy (lbp, base=2) # Entropy With Base 2 For Log of LBP

		statistic_features = np.vstack((contrast,dissimilarity,homogeneity,ASM,energy,correlation)).flatten()
		return (np.concatenate((statistic_features,ent)))

        