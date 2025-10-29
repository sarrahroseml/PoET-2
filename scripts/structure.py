import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt

df = dd.read_parquet(
    '/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/valid.parquet',
    columns=["cluster_representative", "sequence_accession", "structural_accession"]
)

unique_structures = list(set(df.iloc[:,2].str[6:].compute().tolist()))
unique_databases = set([x[:3] for x in unique_structures])

# alphafold 
#.split('-')[1]
af_list = [x[3:] for x in unique_structures if x[:3] == "AF2"]
with open ("/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/af_acc.txt", "w") as fout: 
    for a in af_list: 
        fout.write(a + "\n")

print(af_list)

# count number of sequences per cluster representative
#grouped = df.groupby("cluster_representative").member_accession.size()
#cluster_sizes = grouped.compute()

#sns.histplot(cluster_sizes, bins=50)
#plt.xlabel("Cluster size (# of sequences)")
#plt.ylabel("Count")
#plt.title("Cluster size distribution")
#plt.show()
#plt.savefig("/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/invalid.png")
