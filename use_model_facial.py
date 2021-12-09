from numpy import mod
from scikit_model import Model
import joblib

# csv_list = ["cleaned_on_chin.csv", "cleaned_proper_wear_1min.csv",
#             "cleaned_proper_wear_but_close.csv", "cleaned_under_nose_1min.csv"]
#csv_list = ['cleaned_proper_2k.csv', 'cleaned_improper_2k.csv']
# csv_list = ['./raw_data/cleaned_10k_proper.csv', './raw_data/cleaned_10k_improper.csv']
csv_list = [
    "cleaned/cleaned_rishik_chew.csv",
    "cleaned/cleaned_rishik_no_action.csv",
    "cleaned/cleaned_rishik_smile.csv",
    "cleaned/cleaned_rishik_talk.csv",
    "cleaned/cleaned_sunil_chew.csv",
    "cleaned/cleaned_sunil_chew1.csv",
    "cleaned/cleaned_sunil_no_action.csv",
    "cleaned/cleaned_sunil_no_action1.csv",
    "cleaned/cleaned_sunil_smile.csv",
    "cleaned/cleaned_sunil_smile1.csv",
    "cleaned/cleaned_sunil_talk.csv",
    "cleaned/cleaned_sunil_talk1.csv"
]

window_size = 40

model_obj = Model(csv_list=csv_list, windowSize=window_size)

knn =  model_obj.create_knn_model(n_neighbors=3)
model_obj.k_fold_validation(model_obj.knn, 'knn')

nb = model_obj.create_nb()
model_obj.k_fold_validation(model_obj.nb, 'nb')

rf = model_obj.create_rf()
model_obj.k_fold_validation(model_obj.rf, 'rf')


joblib.dump(rf, 'my_model_facial.pkl', compress=9)
# model_obj.knn_k_plot()
