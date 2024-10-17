import mlflow
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

class ModelTrainer:
    
    def __init__(self,experiment_name,train_set,test_set=None):
        self.train_set = train_set
        self.test_set = test_set
        self.exp_name = experiment_name
        mlflow.set_experiment(self.exp_name)
        
    def train_h2o_glm(self,prev_commit_hash,params=None):
        
        h2o_logistic_model = H2OGeneralizedLinearEstimator(family='binomial') \
                            .train(x=self.train_set.drop("reordered").columns,y='reordered',training_frame=self.train_set)
        
        # log the important stuffs
        with mlflow.start_run():
            
            mlflow.h2o.log_model(h2o_logistic_model,"h2o_logistic_model")
            
            # log perameters
            mlflow.log_param("family","binomial")
            mlflow.log_param("alpha",h2o_logistic_model.get_params()['alpha'])
            mlflow.log_param("lambda",h2o_logistic_model.get_params()['lambda'])
                
            if self.test_set != None:  
                
                preds = h2o_logistic_model.predict(self.test_set).as_data_frame()['p1']
                pred_logits = [1 if pred >= 0.5 else 0 for pred in preds]
                y_true = test_data['reordered'].as_data_frame()['reordered']
                
                #log metrics 
                mlflow.log_metric("precision",precision_score(y_true,pred_logits))
                mlflow.log_metric("recall",recall_score(y_true,pred_logits))
                mlflow.log_metric("f1",f1_score(y_true,pred_logits))
                mlflow.log_metric("AUC",roc_auc_score(y_true,preds))
                mlflow.log_metric("logloss",log_loss(y_true,preds))
                
            # log script url with version 
            commit_url = "https://github.com/d-sutariya/instacart_next_basket_prediction/tree/" + prev_commit_hash
            mlflow.log_param("file url",commit_url)

            # log environment 
            os.system("conda env export > conda.yaml")
            mlflow.log_artifact("conda.yaml")

            # log dataset 
            dataset_path = "https://www.kaggle.com/datasets/deepsutariya/instacart-exp-data" 
            mlflow.log_param("dataset_path",dataset_path)
            
            del y_true
            del preds 
        
            
        return h2o_logistic_model
    
    def train_xgb_rf(self,prev_commit_hash,params=None):
        pass 