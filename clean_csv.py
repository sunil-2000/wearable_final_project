import pandas as pd


# csv_list = ["on_chin.csv", "proper_wear_1min.csv",
# "proper_wear_but_close.csv", "under_nose_1min.csv"]
# csv_list = ['10k_proper.csv', '10k_improper.csv']
# csv_list = [
#     "raw_data_updated/rishik_normal_notalk.csv",
#     "raw_data_updated/rishik_normal2.csv",
#     "raw_data_updated/rishik_underchin.csv",
#     "raw_data_updated/rishik_undernose.csv",
#     "raw_data_updated/sunil_normal.csv",
#     "raw_data_updated/sunil_normal2.csv",
#     "raw_data_updated/sunil_underchin.csv",
#     "raw_data_updated/sunil_undernose.csv",
# # ]
# csv_list = [
#     "facial_classification/rishik_chew.csv",
#     "facial_classification/rishik_no_action.csv",
#     "facial_classification/rishik_smile.csv",
#     "facial_classification/rishik_talk.csv",
#     "facial_classification/sunil_chew.csv",
#     "facial_classification/sunil_no_action.csv",
#     "facial_classification/sunil_smile.csv",
#     "facial_classification/sunil_talk.csv",
#]
# csv_list = [
#     "updated_binary/suniL_correct3.csv",
#     "updated_binary/sunil_correct4.csv",
#     "updated_binary/sunil_incorrect1.csv",
#     "updated_binary/sunil_incorrect4.csv",
#     "updated_binary/sunil_incorrect5.csv",
# ]
csv_list = ["facial_classification/sunil_talk1.csv",
            "facial_classification/sunil_chew1.csv",
            "facial_classification/sunil_no_action1.csv",
            "facial_classification/sunil_smile1.csv"]
for csv in csv_list:
    cols = ["nose", "chin", "left", "right", "label"]
    df = pd.read_csv(csv, names=cols)
    df = df.fillna(0)
    new_df = "cleaned_"+ csv[csv.index("/") + 1 :]
    df.to_csv(new_df, header=False, index=False)
    print(df.head())
