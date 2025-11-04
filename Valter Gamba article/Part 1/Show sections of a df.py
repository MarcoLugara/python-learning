def setup_view():
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)  # Set the display width to a large number
    return pd

pd = setup_view()
df = pd.read_csv('FILENAME')
print(df.iloc[:, :])