def remove_duplicate_speeches(df):
    """
    Remove (row-wise) duplicate speeches if written, i.e. for each written speech track if duplicate
    and only keep the first occurrence
    Remove speeches of speakers without party

    Information loss: drops ??? rows (~ ?? %) with (written) duplicate speech/text
    -> Leaves us with ??? remaining duplicate speeches (non-written)
    """

    # initial_rows_count = df.shape[0]

    # Identify rows where 'text' is a subsequent duplicate (i.e., not the first occurrence)
    is_subsequent_duplicate_text = df['text'].duplicated(keep='first')

    print("Removed duplicate speeches")
    return df[~is_subsequent_duplicate_text]
    # Identify rows where 'written' is True
    # is_written_true = df['written'] == True

    # Combine conditions to find rows to omit: subsequent duplicate text AND written is True
    # rows_to_omit_mask = is_subsequent_duplicate_text & is_written_true

    # Create the new DataFrame by keeping rows that do NOT match the omission criteria
    # df_filtered_by_written_duplicates = df[~rows_to_omit_mask]

    # final_rows_count = df_filtered_by_written_duplicates.shape[0]
    # rows_removed_by_condition = initial_rows_count - final_rows_count

    # print(f"Initial number of rows in df_party_speeches: {initial_rows_count}")
    # print(f"Number of rows after filtering (subsequent duplicate text & written=True): {final_rows_count}")
    # print(f"Number (& Percentage) of rows removed: {rows_removed_by_condition} ({'%.2f' % (rows_removed_by_condition / initial_rows_count)})")


    # return df_filtered_by_written_duplicates