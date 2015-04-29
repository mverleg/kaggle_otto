
"""
	From http://stats.stackexchange.com/questions/28593/mahalanobis-distance-distribution-of-multivariate-normally-distributed-points
"""


#imports and definitions
import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
from utils.loading import get_training_data

chi2 = stats.chi2
np.random.seed(111)


all_data, all_labels = get_training_data()[:2]

#covariance matrix: X and Y are normally distributed with std of 1
#and are independent one of another
covCircle = np.array([[1, 0.], [0., 1.]])
circle = np.random.multivariate_normal([0, 0], covCircle, 1000) #1000 points around [0, 0]
print circle
mahalanobis = lambda p: distance.mahalanobis(p, [0, 0], covCircle.T)
d = np.array(map(mahalanobis, circle)) #Mahalanobis distance values for the 1000 points
d2 = d ** 2 #MD squared

degrees_of_freedom = 2

x = range( len( d2 ))

plt.subplot(111)

plt.scatter( x, d2 )

plt.hlines( chi2.ppf(0.95, degrees_of_freedom), 0, len(d2), label ="95% $\chi^2$ quantile", linestyles = "solid" )
plt.hlines( chi2.ppf(0.975, degrees_of_freedom), 0, len(d2), label ="97.5% $\chi^2$ quantile", linestyles="dashed" )
plt.hlines( chi2.ppf(0.99, degrees_of_freedom), 0, len(d2), label ="99.5% $\chi^2$ quantile", linestyles = "dotted" )

plt.legend()
plt.ylabel("recorded value")
plt.xlabel("observation")
plt.title( 'Detection of outliers at different $\chi^2$ quantiles' )

plt.show()