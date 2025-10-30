import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle 
import subprocess

df = dd.read_parquet(
    '/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/valid.parquet',
    columns=["cluster_representative", "sequence_accession", "structural_accession"]
)

unique_structures = list(set(df.iloc[:,2].str[6:].compute().tolist()))
viral_db = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/pdb_files/viro3d/"
NOM = [x[6:]  for x in unique_structures if x[:3] == "V3D"]
my_list = []
old_db = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/pdb_files/viro3d"
new_db = "/n/groups/marks/databases/foldseek/pdb/viro3d/"
with open("/n/groups/marks/projects/viral_plm/models/PoET-2/data/" + "viro3d.txt", "w") as fout: 
    for n in NOM:
        subprocess.run(f"cp {old_db}/{n}.pdb {new_db}{n}.pdb", shell=True, text=True)

       

#unique_structures = list(set(df.iloc[:,2].str[6:].compute().tolist()))
#unique_databases = set([x[:3] for x in unique_structures])

# alphafold 
#.split('-')[1]


# count number of sequences per cluster representative
#grouped = df.groupby("cluster_representative").member_accession.size()
#cluster_sizes = grouped.compute()

#sns.histplot(cluster_sizes, bins=50)
#plt.xlabel("Cluster size (# of sequences)")
#plt.ylabel("Count")
#plt.title("Cluster size distribution")
#plt.show()
#plt.savefig("/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/invalid.png")
