def print_dataframe_statistics(df):
    print("DataFrame Statistics")
    print("====================\n")

    # Shape of the DataFrame
    print("Shape of the DataFrame:")
    print(df.shape)
    print("\n")

    # Data types of each column
    print("Data types of each column:")
    print(df.dtypes)
    print("\n")

    # General statistics
    print("General statistics (numeric columns):")
    print(df.describe())
    print("\n")

    # Checking for missing values
    print("Missing values in each column:")
    print(df.isnull().sum())
    print("\n")

    # Checking for unique values
    print("Number of unique values in each column:")
    print(df.nunique())
    print("\n")

