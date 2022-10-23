# import useful library
import numpy as np
from scipy.stats import shapiro
from numpy.random import randn
  
# Create data
gfg_data = randn(5000)
  
# conduct the  Shapiro-Wilk Test
print(shapiro(gfg_data))