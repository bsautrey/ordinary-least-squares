Written by Ben Autrey: https://github.com/bsautrey

---Overview---

Implement ordinary least squares from Andrew Ng's CS229 course: http://cs229.stanford.edu/notes/cs229-notes1.pdf. Stochastic gradient descent is used to learn the parameters, i.e. minimize the cost function.

alpha - The learning rate.
dampen - Factor by which alpha is dampened on each iteration. Default is no dampening, i.e. dampen = 1.0
tol - The stopping criteria.
theta - The parameters to be learned.

---Requirements---

* numpy: https://docs.scipy.org/doc/numpy/user/install.html
* matplotlib: https://matplotlib.org/users/installing.html

---Example---

1) Change dir to where OLS.py is.

2) Run this in a python terminal:

from OLS import OLS
ols = OLS()
ols.generate_example()

OR

See the function generate_example() in OLS.py.