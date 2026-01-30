
def remove_non_party_speeches(df): 
    """
    Remove speeches of speakers without party 
        "-": president, ...
        "NI": non-inscrites, non associated members of parliament without party
        "TGI": technical group of independent members, also without party membership
    Information loss: drops 94623 rows without party alignment (roughly 16%)
    """

    # only use speeches where speaker is associated with a party
    parties_to_drop = ["-", "NI", "TGI"]
    # mask_dropped_rows = df["party"].isin(parties_to_drop)

    # n_to_drop = len(df[mask_dropped_rows])
    print(f"Removing speakers without party") #: {n_to_drop} ({'%.2f' % (n_to_drop/len(df))})")
    
    return df[~(df["party"].isin(parties_to_drop))]
