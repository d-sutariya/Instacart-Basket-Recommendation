import os
import time
import csv
import xgboost as xgb
from instacart_model_trainer_script import ModelTrainer

# Function to calculate elapsed time
def get_time(start):
    return time.time() - start

def main():
    # Environment variables
    root_dir = os.getenv('ROOT_DIR', '')  # Ensure ROOT_DIR is set in environment variables
    model_path = os.path.join(root_dir, 'models', 'final_xgb_model.json')
    
    # Get output file path from user input
    output_path = input("Please enter the path for the output file (e.g., cloud path): ")

    dataset_version = "1.2"  # dataset . split method
    model_version = "1.1.1"  # algorithm . param version . used dataset 
    read_as_xgb_dmatrix = True
    train_xgb_gbm = True
    train_xgb_rf = False

    # Load feature names
    train_features_name = []
    with open(os.path.join(root_dir, "final-dataset-generator", "train_set_columns.txt"), "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            train_features_name = row
    train_features_name.remove('')

    test_features_name = []
    with open(os.path.join(root_dir, "final-dataset-generator", "test_set_columns.txt"), "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            test_features_name = row
    test_features_name.remove('')

    train_label_index = train_features_name.index('reordered')

    # Modify features names in test set
    for i in range(len(test_features_name)):
        if test_features_name[i] == 'time_mean_dow_count':
            test_features_name[i] = 'total_ord_count_p_dow'
        if test_features_name[i] == 'time_mean_ohod_count':
            test_features_name[i] = 'total_ord_count_p_ohod'
    # Create a Booster object
    booster = xgb.Booster()

    # Load the model from the JSON file
    booster.load_model(model_path)
        # Load model


    # Model parameters
    params = {
        'max_depth': 1,
        'min_child_weight': 389,
        'verbose': -1,
        'gamma': 438,
        'eta': 0.0907183045377987,
        'subsample': 0.10741019033611729,
        'colsample_bytree': 0.336917979846298,
        'colsample_bylevel': 0.118177830385349,
        'lambda': 11,
        'alpha': 89,
        'booster': 'gbtree',
        'tree_method': 'hist',
        'objective': 'binary:logistic'
    }

    if read_as_xgb_dmatrix:
        train_features_name.remove('reordered')
        # test_features_name.remove('reordered')
        dtrain_2 = xgb.DMatrix(os.path.join(root_dir, "final-dataset-generator", "final_prior_train_set.csv", "part-00000-12859daa-f746-4f84-a1f1-4d24e43087a3-c000.csv"), nthread=-1, feature_names=train_features_name)

    # Train the XGBoost GBM model
    if train_xgb_gbm:
        model_trainer = ModelTrainer("final_instacart_training", dtrain_2)
        xgb_gbm = model_trainer.train_xgb_gbm("c391e8337f10ceb5870cb639159539f5e3497fbf", dataset_version, model_version, params)

    # Additional code can be added for other models as needed, such as XGBoost RF, LightGBM, H2O, etc.

    # Ask the user for the output path (e.g., cloud storage or local file)
    print(f"Output will be saved to: {output_path}")

if __name__ == "__main__":
    main()
