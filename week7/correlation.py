import numpy as np
import scipy.stats
sex = np.array([0,0,0,0,0,1,1,1,1,1]) #0 for M; 1 for F
ethnicity = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0]) #0 for native; 1 for non-native
highestdegree = np.array([1,2,1,1,2,2,1,0,2,1]) #0 for none; 1 for high-school; 2 for university
jobtype = np.array([0,0,0,1,1,2,2,1,2,0]) #0 for board; 1 for healthcare; 2 for education)
print("sex and ethnicity: {}".format(scipy.stats.spearmanr(sex, ethnicity)))
print("sex and highestdegree: {}".format(scipy.stats.spearmanr(sex, highestdegree)))
print("sex and jobtype: {}".format(scipy.stats.spearmanr(sex, jobtype)))
