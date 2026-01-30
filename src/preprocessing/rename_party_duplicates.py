def rename_party_duplicates(df): 
    """
    Merge parties that changed over time
    
    Changes party column
    Information loss: old party names are not kept
    Continuities based on: W. Kaiser and J. Mittag, “Seventy years of transnational political groups in the European Parliament,” European Parliamentary Research Service, Jan. 2023.
    URL: https://www.europarl.europa.eu/RegData/etudes/BRIE/2023/757568/EPRS_BRI(2023)757568_EN.pdf
    """
    
    df['party_adj'] = None 

    pse_snd = ['PSE', 'S&D']
    ppe = ['PPE-DE', 'PPE']
    efd = ['EDD', 'IND/DEM', 'EFDD', 'EFD']
    enf_id = ['ENF', 'ID']
    eldr_alde_renew = ['ELDR','ALDE', 'Renew']
    ngl_theleft = ['GUE/NGL','The Left']
    
    # other parties that do not need to be renamed: 
    others = ['Greens/EFA', 'UEN', 'ECR', 'ITS']
    
    # make sure we handle all parties here: 
    valid_party_values = [
        *pse_snd, *ppe, *efd, *enf_id, *eldr_alde_renew, *ngl_theleft, *others,
    ]
    assert df['party'].isin(valid_party_values).all(), f"Invalid party values: {df[~df['party'].isin(valid_party_values)]['party'].unique()}"

    df.loc[df['party'].isin(pse_snd), 'party_adj'] = 'PSE/S&D' # PSE becomes S&D
    df.loc[df['party'].isin(ppe), 'party_adj'] = 'PPE' # PPE-DE' becomes 'PPE'
    df.loc[df['party'].isin(efd), 'party_adj'] = 'EDD/INDDEM/EFD' # 'EDD' becomes 'IND/DEM' becomes 'EFDD' becomes 'EFD'
    df.loc[df['party'].isin(enf_id), 'party_adj'] = 'ENF/ID' # ENF becomes ID in 2019
    df.loc[df['party'].isin(eldr_alde_renew), 'party_adj'] = 'ELDR/ALDE/Renew' # ELDR becomes ALDE becomes Renew
    df.loc[df['party'].isin(ngl_theleft), 'party_adj'] = 'NGL/The Left' # GUE/NGL becomes The Left

    # if it is none of the above, keep original party name
    df.loc[df['party_adj'].isna(), 'party_adj'] = df.loc[df['party_adj'].isna(), 'party']

    df.drop('party', axis=1, inplace=True)
    df = df.rename(columns={'party_adj': 'party'})

    return df 