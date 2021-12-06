from numpy import mod
from scikit_model import Model
import joblib

# csv_list = ["cleaned_on_chin.csv", "cleaned_proper_wear_1min.csv",
#             "cleaned_proper_wear_but_close.csv", "cleaned_under_nose_1min.csv"]
#csv_list = ['cleaned_proper_2k.csv', 'cleaned_improper_2k.csv']
csv_list = ['./raw_data/cleaned_10k_proper.csv', './raw_data/cleaned_10k_improper.csv']

window_size = 5

model_obj = Model(csv_list=csv_list, windowSize=window_size)

model_obj.create_knn_model(n_neighbors=3)
knn = model_obj.k_fold_validation(model_obj.knn)
svm = model_obj.create_svm_model()

joblib.dump(knn, 'my_model.pkl1', compress=9)
# model_obj.knn_k_plot()
