Final wearable project for INFO4120 (article link: https://news.cornell.edu/stories/2021/12/wearables-robotics-highlight-information-science-student-showcase)
Project Report: https://docs.google.com/document/d/19xRLtGBZTNwVS8HP3fu1ptz1ImDSI8YCmDFiX_pkr-o/edit?usp=sharing 

Software to support SmartMask, a wearable that detects and informs users of
improper mask wear and predicts a user's facial expression underneath a mask.

Sensor_signals.ino: This program reads in the capacitance for all the sensors and outputs the data onto the bluetooth channel. The program also reads in input from the bluetooth channel, which is sent by the computer. The program takes that result and changes the voltage for the pin to the vibrator so that it vibrates for one second.

collect_data.py:
This script connects to the microprocessor’s bluetooth serial port, and reads the data line by line. Each data line contains the raw sensor data, which is 4 dimensional. Each line has nose, left cheek, right cheek, and chin capacitance readings, corresponding to the 4 sensors on the mask. This data is written into a csv, with a label being appended to each line of +1 or -1. +1 indicates improper mask wear, and -1 indicates proper mask wear. We collected 10,000 samples each time the script was run (~ 2 minutes).

feature_engineering.py:
In this file, we define a class, Features, to instantiate an object that contains the relevant feature data generated from raw data files. The class converts the raw list of csv files into one numpy matrix, where each row is a feature data point. The most important functions are gen_features and gen_features_test. Gen_features takes a numpy matrix of raw data, and generates feature data points from it. The function divides the raw data into data sets that have a size equal to the window size, and then converts the raw data into a feature vector. For example, the mean of the first dimension (nose capacitance) is one dimension of the vector. Specifically, for each dimension of the raw data, 9 dimensions are generated for the feature vector. Thus, we currently have 36 features generated from a dataset of size window size. Namely, we calculate: mean, std, var, maximum, minimum, minimum & maximum difference, median, count above mean, and count below mean for each raw data dimension. So, if 20,000 raw data points are collected and the window size is 50, then 400 feature data points are generated. 

scikit_model.py:
In this file, we define a class Model to instantiate a model class that can be used to generate different machine learning models, using the scikit-learn implementations. A model object can perform k-fold stratified validation. The stratify parameter enforces the train and test sets to have data points sampled from the dataset distribution. For example, if there are equal proportions of binary labels (+1/-1 = 50%), then the train and test dataset will have equal proportions of these labels. It can reduce overfitting, which can be very detrimental if the train dataset is overwhelmingly composed of one label. To generate a model, call Model_obj.create_<model_name>, which will return a trained classifier. Currently, we have only trained a SVM and KNN classifier. 

clean_csv.py:
We noticed that a small number of rows in the raw data files had missing values. This is a natural aspect of working with raw data collection. To account for this, we replace missing values with 0. Since the amount of missing values is so small, we suspect that this has no effect on the model training. This cleaning process is more in place to prevent other parts of our code from crashing due to NaN values.

use_model.py
In this file we train a classifier by instantiating a Model object with the csv file list and window size as arguments. We then pickle (serialize) the trained classifier, so that we can access the model in a different file for real-time classification. We tested naivebayes, knn, svm, and random forest algorithms. We found that random forest was the strongest model for both mask-wearing and facial expression classification. Below are the results of the model accuracy on the test datasets. For each model, we performed stratified k-fold validation. The first image are the accuracies for the binary classification problem (mask being worn properly vs not) where proper mask wear has label of -1 and improper has label of +1. The second image is the multi classification problem for facial expressions (labels: 0-> smiling, 1 -> neutral/no expression, 2-> chewing, 3-> talking). The total labels print statement can be ignored for the second image. 

feedback_loop.py
Feedback loop is the script we run to monitor mask wearing behavior. We use a sliding window of size 80 with 50% overlap. Every time the window has a size of 80, we use the unpickled model to make a prediction. This prediction is sent to the microprocessor, which emits a vibration if the mask wear is predicted as improper (label of +1). If the mask is predicted as properly worn (label of -1), there is no vibration. Additionally, if the model predicts properly worn, we use a separate, independent model to predict the facial expression of the user. The predictions are appropriately rendered in the GUI and displayed in the console. 

<img src="4120 Poster.png"></img>
