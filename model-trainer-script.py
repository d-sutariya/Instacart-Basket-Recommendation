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
    
    def train_h2o_glm(self, prev_commit_hash, params=None):
        h2o_logistic_model = H2OGeneralizedLinearEstimator(family='binomial') \
                            .train(x=self.train_set.drop("reordered").columns, y='reordered', training_frame=self.train_set)
        
        with mlflow.start_run():
            mlflow.h2o.log_model(h2o_logistic_model, "h2o_logistic_model")
            mlflow.log_param("family", "binomial")
            mlflow.log_param("alpha", h2o_logistic_model.get_params()['alpha'])
            mlflow.log_param("lambda", h2o_logistic_model.get_params()['lambda'])

            preds = h2o_logistic_model.predict(self.test_set).as_data_frame()['p1']
            y_true = self.test_set['reordered']

            self.__log_details(y_true, preds, prev_commit_hash, params)         
            
            del y_true, preds 
            gc.collect()
        
        return h2o_logistic_model

    def train_h2o_gbm(self, prev_commit_hash, params=None):
        if params is not None and "distribution" not in params.keys():
            params['distribution'] = 'bernoulli'
        
        start = time.time()
        h2o_gbm = H2OGradientBoostingEstimator(**params) \
                  .train(x=self.train_set.drop("reordered").columns, y='reordered', training_frame=self.train_set)
        
        duration = time.time() - start
        
        with mlflow.start_run():
            mlflow.h2o.log_model(h2o_gbm, "h2o_gbm_model")
            mlflow.log_params(params if params is not None else {})
            mlflow.log_param("training_time", duration)

            preds = h2o_gbm.predict(self.test_set).as_data_frame()['p1']
            y_true = self.test_set['reordered']

            self.__log_details(y_true, preds, prev_commit_hash, params, h2o_gbm)

        del y_true, preds
        gc.collect()
        return h2o_gbm

    def train_xgb_gbm(self, prev_commit_hash, params=None):
        
        if params is not None:
            if 'booster' not in params.keys():
                params['booster'] = 'gbtree'
            if 'device' not in params.keys():
                params['device'] = 'cuda'
            if 'tree_method' not in params.keys():
                params['tree_method'] = 'hist'
        else:
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'device': 'cuda',
                'tree_method': 'hist'
            }
        
        start = time.time()
        xgb_gbm = xgb.train(params=params, dtrain=self.train_set[0],
                            evals=[(self.test_set, 'eval')],
                            early_stopping_rounds=50,
                            num_boost_round=1000)
        
        if len(self.train_set) > 1:
            for dtrain in self.train_set[1:]:
                xgb_gbm = xgb.train(params=params, 
                                  dtrain=dtrain,
                                    evals=[(self.test_set, 'eval')],
                                    early_stopping_rounds=50,
                                    num_boost_round=1000, 
                                    xgb_model=xgb_gbm)

        duration = time.time() - start

        with mlflow.start_run():
            mlflow.xgboost.log_model(xgb_gbm, "xgb_gbm_model")
            mlflow.log_param("training_time", duration)

            
            preds = xgb_gbm.predict(self.test_set)
            y_true = self.test_set.get_label()
            self.__log_details(y_true, preds, prev_commit_hash, params, xgb_gbm)
            
        del y_true, preds
        gc.collect()
        return xgb_gbm
    
    def train_xgb_rf(self, prev_commit_hash, params=None):
        if params is None:
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'device': 'cuda',
                'tree_method': 'hist',
                'verbosity': -1
            }
    
        start = time.time()
        xgb_rf = xgb.train(params=params, dtrain=self.train_set[0],
                           evals=[(self.test_set, 'eval')],
                           num_boost_round=1)
    
        if len(self.train_set) > 1:
            for dtrain in self.train_set[1:]:
                xgb_rf = xgb.train(params=params,
                                   dtrain=dtrain,
                                   evals=[(self.test_set, 'eval')],
                                   num_boost_round=1,
                                   xgb_model=xgb_rf)
    
        duration = time.time() - start
    
        with mlflow.start_run():
            mlflow.xgboost.log_model(xgb_rf, "xgb_rf_model")
            mlflow.log_param("training_time", duration)
    
            preds = xgb_rf.predict(self.test_set)
            y_true = self.test_set.get_label()
    
            self.__log_details(y_true, preds, prev_commit_hash, params, xgb_rf)
    
        del y_true, preds
        gc.collect()
        return xgb_rf
    
    def train_lgb_gbm(self, prev_commit_hash, params=None):
        if params is None:
            params = {
                'objective': 'binary',
                # 'device': 'cuda'
            }
    
        start = time.time()
        lgb_gbm = lgb.train(params=params,
                            train_set=self.train_set[0],
                            valid_sets=[self.test_set],
                            callbacks = [
                                lgb.early_stopping(stopping_rounds=30)
                            ],
                            num_boost_round=250)
    
        if len(self.train_set) > 1:
            for dtrain in self.train_set[1:]:
                lgb_gbm = lgb.train(params=params,
                                    train_set=dtrain,
                                    valid_sets=[self.test_set],
                                    callbacks = [
                                        lgb.early_stopping(stopping_rounds=30)
                                    ],
                                    num_boost_round=250,
                                    init_model=lgb_gbm)
    
        duration = time.time() - start
    
        with mlflow.start_run():
            mlflow.lightgbm.log_model(lgb_gbm, "lgb_gbm_model")
            mlflow.log_param("training_time", duration)
    
            preds = lgb_gbm.predict(self.test_set.get_data())
            y_true = self.test_set.get_label()
    
            self.__log_details(y_true, preds, prev_commit_hash, params, lgb_gbm)
    
        del y_true, preds
        gc.collect()
        return lgb_gbm