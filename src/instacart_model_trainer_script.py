# %% [code]
import time
import os
import gc
import json
import mlflow
import h2o
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OGradientBoostingEstimator

class ModelTrainer:
    
    def __init__(self, experiment_name, train_set, test_set=None, target_column='reordered'):
        """
        Initializes the ModelTrainer with the experiment name, training set, and optional test set.
        
        Parameters:
        experiment_name (str): The name of the MLflow experiment.
        train_set (H2OFrame): The training dataset.
        test_set (H2OFrame, optional): The test dataset. Defaults to None.
        target_column (str): The name of the target column. Defaults to 'reordered'.
        """
        self.train_set = train_set
        self.test_set = test_set
        self.exp_name = experiment_name
        self.target_column = target_column  # Store the target column name
        mlflow.set_experiment(self.exp_name)

    def __log_details(self, y_true, preds, prev_commit_hash, params, model=None):
        """
        Logs detailed information about the model's performance, environment, and dataset to MLflow.
        
        Parameters:
        y_true (array-like): True values for the target variable.
        preds (array-like): Predicted values.
        prev_commit_hash (str): The commit hash for version control reference.
        params (dict): Model hyperparameters.
        model (object, optional): The trained model. Defaults to None.
        
        Logs:
        - Precision, recall, F1 score, AUC, and log loss.
        - Commit URL, environment, and dataset details.
        """
        try:
            # Log parameters
            if params != None:
                mlflow.log_params(params)
            else:
                mlflow.log_param("params", None)

            # Log metrics
            pred_logits = [1 if pred >= 0.5 else 0 for pred in preds]
            mlflow.log_metric("precision", precision_score(y_true, pred_logits))
            mlflow.log_metric("recall", recall_score(y_true, pred_logits))
            mlflow.log_metric("f1", f1_score(y_true, pred_logits))
            mlflow.log_metric("AUC", roc_auc_score(y_true, preds))
            mlflow.log_metric("logloss", log_loss(y_true, preds))

            # Log script URL with version 
            commit_url = "https://github.com/d-sutariya/instacart_next_basket_prediction/tree/" + prev_commit_hash
            mlflow.log_param("repository url", commit_url)

            # Log environment 
            os.system("conda env export > conda.yaml")
            mlflow.log_artifact("conda.yaml")

            # Log dataset 
            dataset_path = "https://www.kaggle.com/datasets/deepsutariya/instacart-exp-data" 
            mlflow.log_param("dataset url", dataset_path)

            mlflow.end_run()

        except Exception as e:
            raise RuntimeError(f"Error logging model details: {str(e)}")
    
    def train_h2o_glm(self, prev_commit_hash, dataset_version, model_version, params=None):
        """
        Trains a Generalized Linear Model (GLM) using H2O's binomial family.
        
        Parameters:
        prev_commit_hash (str): The commit hash for version control.
        dataset_version (str): The version of the dataset.
        model_version (str): The version of the model.
        params (dict, optional): Hyperparameters for the model. Defaults to None.
        
        Returns:
        h2o_model: The trained H2O GLM model.
        """
        try:
            start = time.time()
            
            h2o_logistic_model = H2OGeneralizedLinearEstimator(family='binomial') \
                                .train(x=self.train_set.drop("reordered").columns, y='reordered', training_frame=self.train_set)
            duration = time.time() - start
            
            with mlflow.start_run():
                mlflow.h2o.log_model(h2o_logistic_model, "h2o_logistic_model")
                mlflow.log_param("family", "binomial")
                mlflow.log_param("alpha", h2o_logistic_model.get_params()['alpha'])
                mlflow.log_param("lambda", h2o_logistic_model.get_params()['lambda'])
                
                mlflow.log_param("training_time", duration)
                mlflow.set_tag("dataset_version", dataset_version)
                mlflow.set_tag("model_version", model_version)
                mlflow.set_tag("algorithm", "h2o_glm")
                
                progress = h2o_logistic_model.scoring_history().to_dict()
                with open("loss_history.json", "w") as f:
                    json.dump(progress, f)
                mlflow.log_artifact("loss_history.json")
                
                if self.test_set != None:
                    preds = h2o_logistic_model.predict(self.test_set).as_data_frame(use_multi_thread=True)['p1']
                    y_true = self.test_set['reordered'].as_data_frame(use_multi_thread=True)
                else:
                    preds = h2o_logistic_model.predict(self.train_set).as_data_frame(use_multi_thread=True)['p1']
                    y_true = self.train_set['reordered'].as_data_frame(use_multi_thread=True)

                self.__log_details(y_true, preds, prev_commit_hash, params)
                
                del y_true, preds
                gc.collect()

            return h2o_logistic_model
        
        except Exception as e:
            raise RuntimeError(f"Error training H2O GLM: {str(e)}")

    def train_h2o_gbm(self, prev_commit_hash, dataset_version, model_version, params=None):
        """
        Trains a Gradient Boosting Machine (GBM) model using H2O's GradientBoostingEstimator.
        
        Parameters:
        prev_commit_hash (str): The commit hash for version control.
        dataset_version (str): The version of the dataset.
        model_version (str): The version of the model.
        params (dict, optional): Hyperparameters for the model. Defaults to None.
        
        Returns:
        h2o_model: The trained H2O GBM model.
        """
        try:
            if params != None and "distribution" not in params.keys():
                params['distribution'] = 'bernoulli'

            if params is None:
                params = {'distribution': 'bernoulli'}
            
            start = time.time()
            
            if self.test_set != None:
                h2o_gbm = H2OGradientBoostingEstimator(**params) \
                          .train(x=self.train_set.drop("reordered").columns,
                                 y='reordered',
                                 training_frame=self.train_set,
                                 validation_frame=self.test_set)
            else:
                h2o_gbm = H2OGradientBoostingEstimator(**params) \
                          .train(x=self.train_set.drop("reordered").columns,
                                 y='reordered',
                                 training_frame=self.train_set)

            duration = time.time() - start
            
            with mlflow.start_run():
                mlflow.h2o.log_model(h2o_gbm, "h2o_gbm_model")
                mlflow.log_params(params if params != None else {})
                mlflow.log_param("training_time", duration)
                mlflow.set_tag("dataset_version", dataset_version)
                mlflow.set_tag("model_version", model_version)
                mlflow.set_tag("algorithm", "h2o_gbm")
                
                progress = h2o_gbm.scoring_history().to_dict()
                with open("loss_history.json", "w") as f:
                    json.dump(progress, f)
                mlflow.log_artifact("loss_history.json")
                
                if self.test_set != None:
                    preds = h2o_gbm.predict(self.test_set).as_data_frame(use_multi_thread=True)['p1']
                    y_true = self.test_set['reordered'].as_data_frame(use_multi_thread=True)
                else:
                    preds = h2o_gbm.predict(self.train_set).as_data_frame(use_multi_thread=True)['p1']
                    y_true = self.train_set['reordered'].as_data_frame(use_multi_thread=True)

                self.__log_details(y_true, preds, prev_commit_hash, params, h2o_gbm)

            del y_true, preds
            gc.collect()
            return h2o_gbm
        
        except Exception as e:
            raise RuntimeError(f"Error training H2O GBM: {str(e)}")

    def train_xgb_gbm(self, prev_commit_hash, dataset_version, model_version, params=None):
        """
        Trains a Gradient Boosting Machine (GBM) using XGBoost.
        
        Parameters:
        prev_commit_hash (str): The commit hash for version control.
        dataset_version (str): The version of the dataset.
        model_version (str): The version of the model.
        params (dict, optional): Hyperparameters for the model. Defaults to None.
        
        Returns:
        xgb_model: The trained XGBoost GBM model.
        """
        try:
            if params != None:
                if 'booster' not in params.keys():
                    params['booster'] = 'gbtree'
                if 'tree_method' not in params.keys():
                    params['tree_method'] = 'hist'
                if 'objective' not in params.keys():
                    params['objective'] = 'binary:logistic'
                if 'eval_metric' not in params.keys():
                    params['eval_metric'] = 'logloss'

            start = time.time()

            if self.test_set != None:
                watchlist = [(self.train_set, 'train'), (self.test_set, 'eval')]
                xgb_model = xgb.train(params, self.train_set, num_boost_round=500, early_stopping_rounds=30, evals=watchlist)
            else:
                xgb_model = xgb.train(params, self.train_set, num_boost_round=500, early_stopping_rounds=30,evals=[(self.train_set,'train')])

            duration = time.time() - start

            with mlflow.start_run():
                
                mlflow.xgboost.log_model(xgb_model, "xgb_gbm_model")
                mlflow.log_params(params if params != None else {})
                mlflow.log_param("training_time", duration)
                mlflow.set_tag("dataset_version", dataset_version)
                mlflow.set_tag("model_version", model_version)
                mlflow.set_tag("algorithm", "xgb_gbm")

                if self.test_set != None:
                    preds = xgb_model.predict(self.test_set)
                    y_true = self.test_set.get_label()
                else:
                    preds = xgb_model.predict(self.train_set)
                    y_true = self.train_set.get_label()

                self.__log_details(y_true, preds, prev_commit_hash, params, xgb_model)

            del y_true, preds
            gc.collect()
            return xgb_model
        
        except Exception as e:
            raise RuntimeError(f"Error training XGBoost GBM: {str(e)}")

    def train_xgb_rf(self, prev_commit_hash, dataset_version, model_version, params=None):
        """
        Trains a Random Forest using XGBoost's 'random forest' booster.
        
        Parameters:
        prev_commit_hash (str): The commit hash for version control.
        dataset_version (str): The version of the dataset.
        model_version (str): The version of the model.
        params (dict, optional): Hyperparameters for the model. Defaults to None.
        
        Returns:
        xgb_model: The trained XGBoost Random Forest model.
        """
        try:
            if params != None:
                if 'booster' not in params.keys():
                    params['booster'] = 'gbtree'
                if 'objective' not in params.keys():
                    params['objective'] = 'binary:logistic'
                if 'eval_metric' not in params.keys():
                    params['eval_metric'] = 'logloss'

            start = time.time()
           
            if self.test_set != None:
                watchlist = [(self.train_set, 'train'), (self.test_set, 'eval')]
                xgb_rf_model = xgb.train(params, self.train_set, num_boost_round=500, early_stopping_rounds=30, evals=watchlist)
            else:
                xgb_rf_model = xgb.train(params, self.train_set, num_boost_round=500, early_stopping_rounds=30,evals=[(self.train_set,'train')])

            duration = time.time() - start

            with mlflow.start_run():
                mlflow.xgboost.log_model(xgb_rf_model, "xgb_rf_model")
                mlflow.log_params(params if params != None else {})
                mlflow.log_param("training_time", duration)
                mlflow.set_tag("dataset_version", dataset_version)
                mlflow.set_tag("model_version", model_version)
                mlflow.set_tag("algorithm", "xgb_rf")

                if self.test_set != None:
                    preds = xgb_rf_model.predict(self.test_set)
                    y_true = self.test_set.get_label()
                else:
                    preds = xgb_rf_model.predict(self.train_set)
                    y_true = self.test_set.get_label()

                self.__log_details(y_true, preds, prev_commit_hash, params, xgb_rf_model)

            del y_true, preds
            gc.collect()
            return xgb_rf_model
        
        except Exception as e:
            raise RuntimeError(f"Error training XGBoost RF: {str(e)}")

    def train_lgbm(self, prev_commit_hash, dataset_version, model_version, params=None):
        """
        Trains a LightGBM model.
        
        Parameters:
        prev_commit_hash (str): The commit hash for version control.
        dataset_version (str): The version of the dataset.
        model_version (str): The version of the model.
        params (dict, optional): Hyperparameters for the model. Defaults to None.
        
        Returns:
        lgb_model: The trained LightGBM model.
        """
        try:
            if params != None:
                if 'objective' not in params.keys():
                    params['objective'] = 'binary'

            start = time.time()

            
            if self.test_set != None:
                lgb_model = lgb.train(params, self.train_set, num_boost_round=500, early_stopping_rounds=30, valid_sets=[self.test_set])
            else:
                lgb_model = lgb.train(params, self.train_set, num_boost_round=500, early_stopping_rounds=30, valid_sets=[self.train_set])

            duration = time.time() - start

            with mlflow.start_run():
                
                mlflow.lightgbm.log_model(lgb_model, "lgb_model")
                mlflow.log_params(params if params != None else {})
                mlflow.log_param("training_time", duration)
                mlflow.set_tag("dataset_version", dataset_version)
                mlflow.set_tag("model_version", model_version)
                mlflow.set_tag("algorithm", "lgbm")

            if self.test_set != None:
                
                preds = lgb_model.predict(self.test_set.get_data())
                y_true = self.test_set.get_label()
    
                self.__log_details(y_true, preds, prev_commit_hash, params, lgb_gbm)
            else:
                
                preds = lgb_model.predict(self.train_set.get_data())
                y_true = self.train_set.get_label()
    
                self.__log_details(y_true, preds, prev_commit_hash, params, lgb_gbm)

            del y_true, preds
            gc.collect()
            return lgb_model
        
        except Exception as e:
            raise RuntimeError(f"Error training LightGBM: {str(e)}")
