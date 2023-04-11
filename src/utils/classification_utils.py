def list_difference(a, b):
    # Remove list b from list a and preserve order
    return [x for x in a if x not in b]


# In some cases synthetic dataframe doesn't contain all values for the column, so we need to add empty columns to the
# synthetic dataframe for classification
def compare_and_fill_missing_columns(test_df, train_df):
    train_df_copy = train_df.copy()

    # Check that both have equal columns
    if not set(list(train_df_copy.columns.values)).symmetric_difference(
            set(list(test_df.columns.values))) == set():

        columns_missing = list_difference(list(test_df.columns.values), list(train_df_copy.columns.values))

        # Add missing columns to train_df in correct order to prevent index errors
        for column_name in columns_missing:
            column_index = test_df.columns.get_loc(column_name)
            # Replace missing column with a column of zeros to correct index
            train_df_copy.insert(column_index, column_name, 0)

    assert list(train_df_copy.columns.values) == list(test_df.columns.values)

    return train_df_copy
