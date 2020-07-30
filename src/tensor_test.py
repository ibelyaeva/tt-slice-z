import tensorly as tl
import numpy as np


tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)))
unfolded = tl.unfold(tensor, mode=0)
tl.fold(unfolded, mode=0, shape=tensor.shape)

from tensorly.decomposition import tucker
# Apply Tucker decomposition
core, factors = tucker(tensor, rank=[2, 2, 2])
# Reconstruct the full tensor from the decomposed form
tl.tucker_to_tensor(core, factors)