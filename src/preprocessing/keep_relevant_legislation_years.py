
def keep_relevant_legislation_years(df, keep_periods=[8, 9]): 
    print("Kept only speeches from legislation periods", keep_periods)
    return df[df["period"].isin(keep_periods)]