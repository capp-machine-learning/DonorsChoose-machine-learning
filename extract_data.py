'''
Functions for reading the data.

Si Young Byun
'''
import os
import pandas as pd

# Reading the data
def read_data(filename):
    '''
    Read a dataset and print a short summary of the data.
    Return a dataframe of the dataset.
    Input:
    - filename: (string) the directory of the dataset
    Output:
    - df: (pandas dataframe) dataframe of the dataset
    '''
    _, ext = os.path.splitext(filename)

    if ext == '.csv':
        df = pd.read_csv(filename, index_col=0)
    elif ext == '.xls':
        df = pd.read_excel(filename, header=1)
    
    return df

# Summarize the loaded data
def summarize_data(df):

    print("################################################################\n")
    print("Summary for the loaded dataset\n")
    print("Data Shape: {}\n".format(df.shape))
    print("Descritive Statistics:\n\n{}\n".format(df.describe()))
    print("################################################################\n")


if __name__ == "__main__":

    try:
        df = read_data(DATA_DIR + DATAFILE)
        convert_dtypes(df)
        print("\nThe data is successfully loaded!\n")
        summarize_data(df)

        try:
            clean_name = "clean_" + DATAFILE
            directory = DATA_DIR + clean_name
            df.to_csv(DATA_DIR + clean_name)
            print("The loaded data is saved as {}.\n".format(directory))

        except:
            print("Failed to save the dataset.\n")

    except:
        print("Failed to read the data. Please check the filename.")
