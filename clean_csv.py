import pandas as pd


csv_list = ["on_chin.csv", "proper_wear_1min.csv",
            "proper_wear_but_close.csv", "under_nose_1min.csv"]
for csv in csv_list:
    cols = ['nose', 'chin', 'left', 'right', 'label']
    df = pd.read_csv(csv, names=cols)
    df = df.fillna(0)
    new_df = 'cleaned_'+csv
    df.to_csv(new_df, header=False, index=False)
    print(df.head())
