import dask.dataframe as dd

# load only needed columns if possible
df = dd.read_parquet(
    "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/parquet_files"
)
# make sure names are stable
df = df.rename(columns={df.columns[0]: "rep", df.columns[1]: "member"})

#add struct flag once
df = df.assign(is_struct=df["member"].str.contains("SARRAH"))

# drop duplicate (rep, member) pairs so counting is correct
dedup = df.drop_duplicates(subset=["rep", "member"])

# per-cluster counts
counts = dedup.groupby("rep").size().rename("n_total")
n_struct = dedup[dedup["is_struct"]].groupby("rep").size().rename("n_struct")

# join them
agg = dd.merge(counts.to_frame(), n_struct.to_frame(), on="rep", how="left")
agg = agg.fillna({"n_struct": 0})
agg = agg.assign(n_seq=agg["n_total"] - agg["n_struct"])

def classify(row):
    if row.n_struct > 1 and row.n_seq > 1:
        return "finished"
    elif row.n_seq > 1:
        return "not finished"
    else:
        return "invalid"

agg = agg.map_partitions(
    lambda pdf: pdf.assign(cluster_type=pdf.apply(classify, axis=1))
)

finished = agg[agg["cluster_type"] == "finished"]
not_finished = agg[agg["cluster_type"] == "not finished"]

finished.repartition(npartitions=1).to_parquet(
    "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/finished.parquet",
    write_index=True,
)
not_finished.repartition(npartitions=1).to_parquet(
    "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/not_finished.parquet",
    write_index=True,
)
