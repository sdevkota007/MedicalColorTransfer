import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve



def wlsFilter(IN_img, L_img, _lambda=1.0, alpha=1.2):
    smallNum = 0.0001

    r, c = IN_img.shape
    k = r * c

    dy = np.diff(L_img, 1, 0)

    dy = -_lambda / (np.abs(dy) ** alpha + smallNum)
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='constant')      #padding with 1 row of zeros at the end
    dy = dy.flatten()

    dx = np.diff(L_img, 1, 1)

    dx = -_lambda / (np.abs(dx) ** alpha + smallNum)
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='constant')      #padding with 1 column of zeros at the end
    dx = dx.flatten()

    B = np.zeros(shape=(dx.shape[0], 2))
    B[:, 0] = dx
    B[:, 1] = dy

    A = spdiags(B.T, diags=[-r,-1], m=k, n=k)

    e = dx
    w = np.pad(dx, ((r, 0)), mode='constant')        #padding 0 r times at the beginning of array
    w = w[0:-r]

    s = dy
    n = np.pad(dy, ((1, 0)), mode='constant')         #padding 0 one time at the beginnig of array
    n = n[0:-1]

    D = 1 - (e + w + s + n)
    A = A + A.T + spdiags(D.T, diags=0, m=k, n=k)
    OUT = spsolve(A, IN_img.flatten())

    OUT = np.reshape(OUT, (r, c))

    return OUT


row, col = 4, 4
IN = np.hstack([np.ones((4,2)), np.full((4,2), 0.0001)])

L = np.log(IN)
print(IN)
print(L)

out = wlsFilter(IN, L)