import numpy as np 

thing = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/structure_arrays/7e9t_C.npz"

data = np.load(thing)
print(data["sequence"])
