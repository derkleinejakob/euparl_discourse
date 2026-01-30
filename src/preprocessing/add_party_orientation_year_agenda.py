def add_party_orientation_year_agenda(df): 
    """
    Create year and unique agenda identifier
    
    Create broader party blocks
    
    Creates new column "block" with values in (left, greens, social_democratic, christian_conservative, liberal, right_populist)
    No information loss here
    party orientation based on: W. Kaiser and J. Mittag, “Seventy years of transnational political groups in the European Parliament,” European Parliamentary Research Service, Jan. 2023.
    URL: https://www.europarl.europa.eu/RegData/etudes/BRIE/2023/757568/EPRS_BRI(2023)757568_EN.pdf

    full labels: 
                - Left and (Post-) Communist groups
                - Green-, alternative-, region-oriented groups
                - Socialist/Social democratic groups
                - Christian-democrat and Conservative groups
                - Liberal groups
                - National conservative, eurosceptic, right and extreme right groups
    """

    # add year and unique agenda identifier
    df["year"] = df.apply(lambda s: int(s["date"][:4]), axis=1)
    df["agenda"] = df["agenda"]+df["date"]

    df['block'] = None 

    # left
    df.loc[df['party'].isin(['GUE/NGL','The Left']), 'block'] = 'left'
    # green
    df.loc[df['party'].isin(['Greens/EFA']), 'block'] = 'green'
    # social democratic
    df.loc[df['party'].isin(['PSE', 'S&D']), 'block'] = 'social_democratic'
    # christian conservative
    df.loc[df['party'].isin(['PPE-DE', 'PPE']), 'block'] = 'christian_conservative'
    # liberal
    df.loc[df['party'].isin(['ELDR','ALDE', 'Renew']), 'block'] = 'liberal'
    # right populist
    df.loc[df['party'].isin(['EFDD', 'EFD','ITS', 'ENF', 'ID', 'IND/DEM', 'ECR', 'UEN', 'EDD', 'ITS']), 'block'] = '(extreme)_right'
    
    return df