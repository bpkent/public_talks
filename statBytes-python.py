# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Getting started with Python for statistical computing

# <headingcell level=4>

# What is Python?

# <markdowncell>

# + A general purpose, high-level programming language
# + Design, syntax, and culture promote human readability
# + Interpreted
# + Dynamic typing
# + Automatic memory management
# + Support the object-oriented paradigm

# <headingcell level=4>

# The statistical computing "stack"

# <markdowncell>

# + *IPython* is an incredibly useful tool for interactive computing (like the R command line).
# + Typically, write code in a text editor, then use the *run* command in IPython.
# + Play with individual blocks of code interactively at the IPython prompt.
# + Not all of the following libraries are always necessary.
# + You don't have to import the whole package.

# <codecell>

import numpy as np     ## vector and matrix operations
import scipy as sp     ## grab-bag of statistical and science tools
import matplotlib.pyplot as plt     ## matplotlib - plots
import pandas as pd     ## emulates R data frames
import statsmodels.api as sm     ## scikits.statsmodels - statistics library
import patsy    ## emulates R model formulas

import sklearn as skl     ## scikits.learn - machine learning library
from sklearn import mixture as sklmix

# <headingcell level=4>

# Installation

# <markdowncell>

# *Linux (Ubuntu)*
# 
# + Python should be included in the OS. To install new packages use  
# `(sudo) pip install [package]`.
# 
# *Mac*
# 
# + Python should be installed already. Consider a bundled installation to get all the packages easier.
# 
# *Windows*
# 
# + Check out the bundled scientific python distributions.
#   + [Enthought](https://www.enthought.com/products/canopy/academic/) (free for academic use)
#   + [Anaconda](https://store.continuum.io/) (Continuum Analytics)
#   + [Python(x,y)](https://code.google.com/p/pythonxy/)

# <headingcell level=4>

# Generate some random variables

# <codecell>

x = np.linspace(0, 10, 100)
y = np.random.normal(loc=x, scale=1.0)
plt.plot(x, y, 'bo')

# <headingcell level=4>

# Basic linear regression

# <codecell>

ols_model = sm.OLS(y, x)
ols_fit = ols_model.fit()
yhat = ols_fit.predict()

print ols_fit.summary()
plt.plot(x, y, 'bo', x, yhat, 'r-', lw=2)

# <headingcell level=4>

# Open a CSV file with Pandas

# <codecell>

iris = pd.read_csv('iris.csv')
print iris, '\n'
print iris.head(10)

# <headingcell level=4>

# Compute summary statistics

# <codecell>

print np.round(iris.describe(), 2)

# <codecell>

type_grp = iris.groupby('Type')
print "number of groups:", type_grp.ngroups, "\n"
print "maximums:\n", type_grp.max(), "\n"
print "size of each group:\n", type_grp.size(), "\n"

# <headingcell level=4>

# Plot the data

# <codecell>

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iris['PW'], iris['PL'], iris['SW'], c=iris['Type'])

ax.set_xlabel('PW')
ax.set_ylabel('PL')
ax.set_zlabel('SW')
ax.azim = 140
ax.elev = 15

# <headingcell level=4>

# Check the state of the namespace

# <codecell>

whos

# <headingcell level=4>

# Get help for a function

# <codecell>

# %pdoc sklmix.GMM
# %psource sklmix.GMM

#sklmix.GMM?
#sklmix.GMM??

# <headingcell level=4>

# Pick your favorite clustering algorithm

# <codecell>

gmm_model = sklmix.GMM(n_components=3, covariance_type='full')
gmm_model.fit(iris[['PW', 'PL', 'SW']])
yhat = gmm_model.predict(iris[['PW', 'PL', 'SW']])
crosstab = pd.crosstab(iris['Type'], yhat, rownames=['true'], colnames=['predicted'])
print crosstab

# <headingcell level=4>

# Align the confusion matrix with a non-standard package

# <codecell>

import munkres
import sys
m = munkres.Munkres()
cost = munkres.make_cost_matrix(crosstab.values.tolist(), lambda x : sys.maxint - x)
align = m.compute(cost)
print align, '\n'

permute = [x[1] for x in align]
new_label = np.argsort(permute)
yhat_new = new_label[yhat]
print pd.crosstab(iris['Type'], yhat_new, rownames=['true'], colnames=['predicted'])

# <headingcell level=4>

# Bridging the gap with Rpy2

# <codecell>

from rpy2.robjects import r
from rpy2.robjects.numpy2ri import numpy2ri as np2r

Xr = np2r(iris[['PW', 'PL', 'SW']].values)
d = r.dist(Xr)
tree = r.hclust(d, method='ward')
yhat_hclust = r.cutree(tree, k=3)

print pd.crosstab(iris['Type'], yhat_hclust, rownames=['true'], colnames=['predicted'])

# <headingcell level=4>

# Using non-base packages in Rpy2

# <codecell>

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
r = robjects.r

e1071 = importr('e1071')
Yr = np2r(iris['Type'])
Yr = r.factor(Yr)
svm = e1071.svm(Xr, Yr)
yhat = r.predict(svm, Xr)
print r.table(yhat, Yr)

# <headingcell level=4>

# ggplot2 in python with Rpy2

# <markdowncell>

# Thanks to [Fei Yu](http://www.thefeiyu.com/) for this vignette.

# <codecell>

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

r = robjects.r
r.library("ggplot2")
rnorm = r["rnorm"]

dtf = robjects.DataFrame({'x': rnorm(300, mean=0) + rnorm(100, mean=3),
                          'y': rnorm(300, mean=0) + rnorm(100, mean=3)})
robjects.globalenv['dtf'] = dtf    # assign to R's environment
r("gp = ggplot(dtf, aes(x, y))")
r("pp = gp + geom_point(size=0.6)")
# r("plot(pp)")    # I can't get this to work on my system, but saving the plot is just as good.
r("ggsave(filename='test.png', plot=pp, scale=0.3)")

from IPython.core.display import Image 
Image(filename='test.png') 

# <headingcell level=4>

# Python resources

# <markdowncell>

# + [Python tutorial](http://docs.python.org/2/tutorial/)
# + [Dive into Python](http://www.diveintopython.net/toc/index.html) (old but good)
# + [Ipython](http://ipython.org/)
# + [Ipython video](https://www.youtube.com/watch?v=2G5YTlheCbw)
# + [py2exe](http://www.py2exe.org/): compile python to a Windows executable file

# <headingcell level=4>

# Package websites

# <markdowncell>

# + [NumPy](http://www.numpy.org/)
# + [SciPy](http://www.scipy.org/)
# + [matplotlib](http://matplotlib.org/)
# + [pandas](http://pandas.pydata.org/)
# + [scikit-learn](http://scikit-learn.org/stable/)
# + [statsmodels](http://statsmodels.sourceforge.net/)
# + [patsy](http://patsy.readthedocs.org/en/latest/index.html): symbolic formulas for models

# <headingcell level=4>

# Rosetta stones

# <markdowncell>

# + [NumPy for MATLAB users, source 1](http://www.scipy.org/NumPy_for_Matlab_Users)
# + [NumPy for MATLAB users, source 2](http://mathesaurus.sourceforge.net/matlab-numpy.html)
# + [NumPy for R users](http://mathesaurus.sourceforge.net/r-numpy.html)
# + [MATLAB vs. R vs. Python](http://hyperpolyglot.org/numerical-analysis)

# <headingcell level=4>

# Miscellaneous

# <markdowncell>

# + [Fernando Perez](https://twitter.com/fperez_org) (IPython)
# + [Wes McKinney](https://twitter.com/wesmckinn) (pandas)
# + [PyData conference videos](http://pyvideo.org/category/18/pydata)
# + [SciPy 2012 conference videos](http://pyvideo.org/category/20/scipy_2012)
# + [Python for Data Analysis](http://www.amazon.com/books/dp/1449319793) (Wes McKinney, 2012)
# + [http://slendrmeans.wordpress.com/](http://slendrmeans.wordpress.com/)

# <headingcell level=4>

# Why Python?

# <markdowncell>

# + Somebody in power told you to.
# + Communication with other people.
# + Communication between parts of a data analysis pipeline.
# + Faster than R (often, but not always).
# + Simple syntax for object-oriented designs.
# + Written by computer scientists and programmers (not statisticians).
# + Much bigger ecosystem of general language tools (but smaller set of statistics tools).
# + Production-quality code.

# <headingcell level=4>

# Ipython Notebook

# <markdowncell>

# Relevant links:
# 
# + http://ipython.org/notebook.html
# + https://github.com/ipython/nbconvert
# + http://nbviewer.ipython.org/
# 
# For Ubuntu:
# 
# + `sudo apt-get install ipython-notebook`
# + `ipython notebook --pylab=inline --script`
# + `nbconvert latex statBytes-python`
# + `pdflatex statBytes-python.tex`
# 
# The `--pylab==inline` flag embeds matplotlib output in the notebook rather than popping up a new window for each figure. The `--script` flat saves a `.py` script along with the `.ipynb` file which is ready to be run in a regular python or ipython prompt.
# 
# There are (at least) two ways to post a notebook online. For this notebook I pushed the .ipynb file to a public github repo, open the file on the github website, clicked the "raw" button, and pasted the URL into the box at the top of the nbviewer website. Pretty straightforward.

