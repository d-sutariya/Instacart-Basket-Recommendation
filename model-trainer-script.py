class ModelTrainer:
    
    def __init__(self, experiment_name, train_set, test_set=None, target_column='reordered'):
        self.train_set = train_set
        self.test_set = test_set
        self.exp_name = experiment_name
        self.target_column = target_column  # Store the target column name
        mlflow.set_experiment(self.exp_name)

    def __log_details(self, y_true, preds, prev_commit_hash, params, model=None):
        # Log parameters
        if params is not None:
            mlflow.log_params(params)
        else:
            mlflow.log_param("params", None)

        if self.test_set is not None:  
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
    
    def train_h2o_glm(self, prev_commit_hash,dataset_version,model_version, params=None):
        start = time.time()
        h2o_logistic_model = H2OGeneralizedLinearEstimator(family='binomial') \
                            .train(x=self.train_set[0].drop("reordered").columns, y='reordered', training_frame=self.train_set[0])
        duration = get_time(start)
        with mlflow.start_run():
            
            mlflow.h2o.log_model(h2o_logistic_model, "h2o_logistic_model")
            mlflow.log_param("family", "binomial")
            mlflow.log_param("alpha", h2o_logistic_model.get_params()['alpha'])
            mlflow.log_param("lambda", h2o_logistic_model.get_params()['lambda'])
            
            mlflow.log_param("training_time", duration)            
            mlflow.set_tag("dataset_version",dataset_version)
            mlflow.set_tag("model_version",model_version)  
            mlflow.set_tag("algorithm","h2o_glm")
            
            progress = h2o_logistic_model.scoring_history().to_dict()
            
            with open("loss_history.json", "w") as f:
                json.dump(progress, f)

            mlflow.log_artifact("loss_history.json")
            preds = h2o_logistic_model.predict(self.test_set).as_data_frame(use_multi_thread=True)['p1']
            y_true = self.test_set['reordered'].as_data_frame(use_multi_thread=True)

            self.__log_details(y_true, preds, prev_commit_hash, params)         
            
            del y_true, preds 
            gc.collect()
        
        return h2o_logistic_model

    def train_h2o_gbm(self, prev_commit_hash,dataset_version,model_version, params=None):
        
        if params is not None and "distribution" not in params.keys():
            params['distribution'] = 'bernoulli'
        
        if params == None:
            params = dict()
            params['distribution'] = 'bernoulli'
        
        start = time.time()
        h2o_gbm = H2OGradientBoostingEstimator(**params) \
                  .train(x=self.train_set[0].drop("reordered").columns,
                         y='reordered',
                          training_frame=self.train_set[0],
                        validation_frame=self.test_set,
                         
                        )
        
        duration = time.time() - start
        
        with mlflow.start_run():
            mlflow.h2o.log_model(h2o_gbm, "h2o_gbm_model")
            mlflow.log_params(params if params is not None else {})
            mlflow.log_param("training_time", duration)            
            mlflow.set_tag("dataset_version",dataset_version)
            mlflow.set_tag("model_version",model_version)  
            mlflow.set_tag("algorithm","h2o_gbm")
            
            progress = h2o_gbm.scoring_history().to_dict()
            with open("loss_history.json", "w") as f:
                json.dump(progress, f)

            mlflow.log_artifact("loss_history.json")
            

            preds = h2o_gbm.predict(self.test_set).as_data_frame(use_multi_thread=True)['p1']
            y_true = self.test_set['reordered'].as_data_frame(use_multi_thread=True)

            self.__log_details(y_true, preds, prev_commit_hash, params, h2o_gbm)

        del y_true, preds
        gc.collect()
        return h2o_gbm

    def train_xgb_gbm(self, prev_commit_hash,dataset_version,model_version,params=None):
        
        if params is not None:
            if 'booster' not in params.keys():
                params['booster'] = 'gbtree'
            if 'device' not in params.keys():
                params['device'] = 'cuda'
            if 'tree_method' not in params.keys():
                params['tree_method'] = 'hist'
            if  'objective' not in params.keys():
                params[ 'objective'] = 'binary:logistic'

        else:
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'device': 'cuda',
                'tree_method': 'hist'
            }
        progress = dict()
        start = time.time()
        xgb_gbm = xgb.train(params=params, dtrain=self.train_set[0],
                           evals=[(self.train_set[0],'train_logloss'),(self.test_set, 'test_logloss')],
                           evals_result= progress,
                            early_stopping_rounds=50,
                            num_boost_round=1000)
        
        if len(self.train_set) > 1:
            for dtrain in self.train_set[1:]:
                xgb_gbm = xgb.train(params=params, 
                                      dtrain=dtrain,
                                       evals=[(dtrain,'train_logloss'),(self.test_set, 'test_logloss')],
                                       evals_result= progress,
                                        early_stopping_rounds=50,
                                        num_boost_round=1000, 
                                        xgb_model=xgb_gbm)

        duration = time.time() - start

        with mlflow.start_run():
            preds = xgb_gbm.predict(self.test_set)
            signature = infer_signature(self.train_set[0].get_data(),preds)
            mlflow.set_tag("dataset_version",dataset_version)
            mlflow.set_tag("model_version",model_version)
            mlflow.set_tag("algorithm","xgb_gbm")
            
            mlflow.xgboost.log_model(xgb_gbm, "xgb_gbm_model",signature=signature)
            mlflow.log_param("training_time", duration)
            
            with open("loss_history.json", "w") as f:
                json.dump(progress, f)

            mlflow.log_artifact("loss_history.json")
            
            
            y_true = self.test_set.get_label()
            self.__log_details(y_true, preds, prev_commit_hash, params, xgb_gbm)
            
        del y_true, preds
        gc.collect()
        return xgb_gbm
    
    def train_xgb_rf(self, prev_commit_hash,dataset_version,model_version, params=None):
        
        if params is None:
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'device': 'cuda',
                'tree_method': 'hist'
            }
        progress = dict()
        start = time.time()
        xgb_rf = xgb.train(params=params, dtrain=self.train_set[0],
                           evals=[(self.train_set[0],'train_logloss'),(self.test_set, 'test_logloss')],
                           evals_result= progress,
                           num_boost_round=1)
    
        if len(self.train_set) > 1:
            for dtrain in self.train_set[1:]:
                xgb_rf = xgb.train(params=params,
                                   dtrain=dtrain,
                                   evals=[(self.train_set,'train_logloss'),(self.test_set, 'test_logloss')],
                                   evals_result= progress,
                                   num_boost_round=1,
                                   xgb_model=xgb_rf)
    
        duration = time.time() - start
    
        with mlflow.start_run():
            
           
            preds = xgb_rf.predict(self.test_set)
            mlflow.log_param("training_time", duration)
            mlflow.set_tag("algorithm","xgb_rf")
            signature = infer_signature(self.train_set[0].get_data(),preds)
            mlflow.xgboost.log_model(xgb_rf, "xgb_rf_model",signature=signature)
            mlflow.set_tag("dataset_version",dataset_version)
            mlflow.set_tag("model_version",model_version)
           
            with open("loss_history.json", "w") as f:
                json.dump(progress, f)

            mlflow.log_artifact("loss_history.json")
            
            y_true = self.test_set.get_label()
    
            self.__log_details(y_true, preds, prev_commit_hash, params, xgb_rf)
    
        del y_true, preds
        gc.collect()
        return xgb_rf
    
    def train_lgb_gbm(self, prev_commit_hash,dataset_version,model_version, params=None):
        if params is None:
            params = {
                'objective': 'binary',
                # 'device': 'cuda'
            }
        progress = dict()
        start = time.time()
        lgb_gbm = lgb.train(params=params,
                            train_set=self.train_set[0],
                            valid_sets=[self.train_set[0],self.test_set],
                            valid_names = ['train_set',"test_set"],
                            callbacks = [
                                lgb.early_stopping(stopping_rounds=30),
                                lgb.record_evaluation(progress)
                            ],
                            num_boost_round=250)
    
        if len(self.train_set) > 1:
            for dtrain in self.train_set[1:]:
                lgb_gbm = lgb.train(params=params,
                                    train_set=dtrain,
                                    valid_sets=[dtraub,self.test_set],
                                    valid_names = ['train_set',"test_set"],
                                    callbacks = [
                                        lgb.early_stopping(stopping_rounds=30),
                                        lgb.record_evaluation(progress)
                                    ],
                                    num_boost_round=250,
                                    init_model=lgb_gbm)
                
        with open("loss_history.json","w") as f:
            json.dump(progress,f)
            
        duration = time.time() - start
    
        with mlflow.start_run():
            
            mlflow.set_tag("dataset_version",dataset_version)
            mlflow.set_tag("model_version",model_version)
            mlflow.set_tag("algorithm","lgb_gbm")
            
            mlflow.lightgbm.log_model(lgb_gbm, "lgb_gbm_model")
            mlflow.log_param("training_time", duration)
            mlflow.log_artifact("loss_history.json")
            preds = lgb_gbm.predict(self.test_set.get_data())
            y_true = self.test_set.get_label()
    
            self.__log_details(y_true, preds, prev_commit_hash, params, lgb_gbm)
    
        del y_true, preds
        gc.collect()
        return lgb_gbm