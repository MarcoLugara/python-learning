def adding_new_Ateco_letters(df_Extra, df):
    df_Extra = pd.read_csv('ATECO_codes.csv')
    df_Extra = df_Extra.rename(columns={
        'Codice Ateco': 'Codice',
        'Titolo Ateco 2007 aggiornamento 2022': 'Codice_desc'}
    )
    df_Extra = df_Extra[['Codice', 'Codice_desc']]

    """
    Basic Syntax for INDEXROW()
    for index, row in dataframe.iterrows():
        # index: the row index
        # row: a Series containing the row data
        # access row data using row['column_name']
    """
    # Getting the values from the ISTAT database related to the general ateco codes
    ateco = list()
    atecoX = list()
    for index, row in df_Extra.iterrows():
        if len(row['Codice']) == 1:
            ateco.append(row['Codice'])
            atecoX.append(row['Codice_desc'])
    print(ateco)
    print(atecoX)

    '''
    #Now, we build a new dataframe to test we don't mess anything up and work on that
    df1 = df.copy()


    # Create mapping dictionary, which maps each n-th value in the first list to the n-th value in the second list
    mapping_dict = dict(zip(ateco, atecoX))

    """ ZIP FUNCTION
    The zip(iterable1, iterable2, iterable3, ...) function takes multiple iterables (lists, tuples, etc.) and returns an iterator of tuples where:
    The i-th tuple contains the i-th element from each of the input iterables
    It stops when the shortest input iterable is exhausted
    It's a perfect match if the length of all iterable1, iterable2, etc... is the same
    """

    # Creating the new 2 columns which have the first 2 characters of ATECO and the related description from the ISTAT DF
    for letter in ateco:

    df1['ateco'] = df_Extra['Codice']

    df['ateco'] = df['ATECO'].astype(str).str[:2]  # .str[:2] takes up to the 2nd character in the string
    df['AtecoX'] = df['Ateco'].map(mapping_dict)

    #Putting the new 2 columns before the ATECO ones
    new_cols = df[['Ateco', 'AtecoX']]
    df = df.drop(df.columns[12:], axis=1)  # cols.remove(thing) works only for one element, for a list. Drop works best for DF for rows and columns. SAY THE AXIS
    #print(df.columns)
    df = pd.concat([df.iloc[:, :1], new_cols, df.iloc[:, 1:]], axis=1)  # axis=0 concatenates rows, axis=1 concatenates columns
    #print(df.columns)

    # Updating the csv and excel
    df.to_csv('Tidier_Dataset.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)
    # print(df.columns)

    # Sorting the DF list by Ateco code, in case it's needed somewhere
    df_sorted_by_Ateco = df.sort_values(['Ateco', 'ATECO'])
            #This will sort first by Ateco and then ONLY WHEN MATCHING, by ATECO
    return df, df_sorted_by_Ateco
    '''
