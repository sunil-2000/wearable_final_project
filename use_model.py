from numpy import mod
from scikit_model import Model
import joblib

# csv_list = ["cleaned_on_chin.csv", "cleaned_proper_wear_1min.csv",
#             "cleaned_proper_wear_but_close.csv", "cleaned_under_nose_1min.csv"]
#csv_list = ['cleaned_proper_2k.csv', 'cleaned_improper_2k.csv']
# csv_list = ['./raw_data/cleaned_10k_proper.csv', './raw_data/cleaned_10k_improper.csv']

csv_list = [
    "cleaned/cleaned_rishik_normal_notalk.csv",
    "cleaned/cleaned_rishik_normal2.csv",
    "cleaned/cleaned_rishik_underchin.csv",
    "cleaned/cleaned_rishik_undernose.csv",
    "cleaned/cleaned_sunil_normal.csv",
    "cleaned/cleaned_sunil_normal2.csv",
    "cleaned/cleaned_sunil_underchin.csv",
    "cleaned/cleaned_sunil_undernose.csv",
    "cleaned/cleaned_suniL_updated_correct.csv",
    "cleaned/cleaned_suniL_updated_correct2.csv",
    "cleaned/cleaned_suniL_updated_undernose.csv",
    "cleaned/cleaned_8:42suniL_correct3.csv",
    "cleaned/cleaned_8:42suniL_correct4.csv",
    "cleaned/cleaned_8:42suniL_incorrect1.csv",
    "cleaned/cleaned_8:42suniL_incorrect4.csv",
    "cleaned/cleaned_8:42suniL_incorrect5.csv"
]#

window_size = 5

model_obj = Model(csv_list=csv_list, windowSize=window_size)

knn =  model_obj.create_knn_model(n_neighbors=3)
model_obj.k_fold_validation(model_obj.knn, 'knn')
svm =model_obj.create_svm_model()
model_obj.k_fold_validation(model_obj.svm, 'svm')
nb = model_obj.create_nb()
model_obj.k_fold_validation(model_obj.nb, 'nb')
rf = model_obj.create_rf()
model_obj.k_fold_validation(model_obj.rf, 'rf')


joblib.dump(rf, 'my_model_binary.pkl', compress=9)
# model_obj.knn_k_plot()
