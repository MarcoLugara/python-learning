import pandas as pd

######################################


# Alternative version that's more dynamic
def create_dataset(input_file, output_file1, output_file2):
    """
    Create a dataset with only 2023 and 2024 data from the original dataset.

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path where the output CSV will be saved
    """

    # Read the original dataset
    df = pd.read_csv(input_file)

    # Remove columns that contain '2022' in their name
    columns_to_keep = [col for col in df.columns if '2022' not in col]

    # Create the new dataframe
    df = df[columns_to_keep].copy()

    # Save to CSV and Excel
    df.to_csv(output_file1, index=False)
    df.to_excel(output_file2, index=False)

    return df

#OLD CODE
'''
# Create the dataset
df = create_dataset('Starting_Dataset.csv', 'Tidier_Dataset.csv', 'Tidier_Dataset.xlsx')

'''

#CODE