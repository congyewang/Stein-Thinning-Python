import numpy as np

import pysteinthin as thin


sample = np.asarray(
    np.random.random(size=(10, 2)),
    dtype=np.float64,
    order='F'
)
grad_sample = -sample

print(thin.thinning(4, sample, grad_sample))
