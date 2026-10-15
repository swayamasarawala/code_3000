import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    keys = ["age", "gender", "zip3"]

    # Missing identifiers cannot establish an exact identity match.
    auxiliary = aux_df.dropna(subset=keys + ["name"])
    # Count records, not distinct names: different people may share a name.
    unique_aux = auxiliary.loc[
        ~auxiliary.duplicated(subset=keys, keep=False)
    ]
    linked = anon_df.dropna(subset=keys).merge(
        unique_aux[keys + ["name"]], on=keys, how="inner",
        validate="many_to_one",
    )
    return linked[["anon_id", "name"]].rename(
        columns={"name": "matched_name"}
    ).reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if anon_df.empty:
        return 0.0
    # Count each matched anonymous record once.
    matched_ids = matches_df.loc[
        matches_df["anon_id"].isin(anon_df["anon_id"]), "anon_id"
    ].nunique()
    return matched_ids / len(anon_df)
