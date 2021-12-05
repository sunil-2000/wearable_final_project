from scikit_model import Model
import joblib

csv_list = ["below_nose(1).csv", "properly_worn.csv", "under_chin.csv"]
window_size = 50

model_obj = Model(csv_list=csv_list, windowSize=window_size)

model_obj.create_knn_model(n_neighbors=3)
knn = model_obj.k_fold_validation(model_obj.knn)

joblib.dump(knn, 'my_model.pkl', compress=9)
model_obj.knn_k_plot()