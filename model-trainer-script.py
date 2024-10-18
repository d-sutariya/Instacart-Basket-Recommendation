class ModelTrainer:
    
    def __init__(self,experiment_name,train_set,test_set=None):
        self.train_set = train_set
        self.test_set = test_set
        self.exp_name = experiment_name
        mlflow.set_experiment(self.exp_name)

    def __log_details(self,y_true,preds,prev_commit_hash,params):
        #log params 
        if params!=None:
                mlflow.log_params(params)
        else:
                mlflow.log_param("params",None)
                
        if self.test_set is not None:  
            
            #log metrics 
            pred_logits = [1 if pred >= 0.5 else 0 for pred in preds]
            mlflow.log_metric("precision",precision_score(y_true,pred_logits))
            mlflow.log_metric("recall",recall_score(y_true,pred_logits))
            mlflow.log_metric("f1",f1_score(y_true,pred_logits))
            mlflow.log_metric("AUC",roc_auc_score(y_true,preds))
            mlflow.log_metric("logloss",log_loss(y_true,preds))
            
        # log script url with version 
        commit_url = "https://github.com/d-sutariya/instacart_next_basket_prediction/tree/" + prev_commit_hash
        mlflow.log_param("repository url",commit_url)

        # log environment 
        os.system("conda env export > conda.yaml")
        mlflow.log_artifact("conda.yaml")

        # log dataset 
        dataset_path = "https://www.kaggle.com/datasets/deepsutariya/instacart-exp-data" 
        mlflow.log_param("dataset url",dataset_path)

        mlflow.end_run()
    
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

            preds = h2o_logistic_model.predict(self.test_set).as_data_frame()['p1']
            y_true = self.test_set['reordered'].as_data_frame()['reordered']

            # log dataset path,script path, environment
            self.__log_details(y_true,preds,prev_commit_hash,params)         
            
            del y_true
            del preds 
        
        return h2o_logistic_model

    def train_h2o_gbm(self,prev_commit_hash,params=None):

        if params != None:
            if "distribution" not in params.keys():
                params['distribution'] = 'bernoulli'
            
            start = time.time()
                
            h2o_gbm = H2OGradientBoostingEstimator(**params) \
                    .train(x=self.train_set.drop("reordered").columns,y='reordered',training_frame = self.train_set)
        else:
            
            start = time.time()
            h2o_gbm = H2OGradientBoostingEstimator() \
                    .train(x=self.train_set.drop("reordered").columns,y='reordered',training_frame = self.train_set)
            
        duration = get_time(start)
        
        with mlflow.start_run():
            # log model
            mlflow.h2o.log_model(h2o_gbm,"h2o_gbm_model")
            
            # log params
            if params!=None:
                mlflow.log_params(params)
            else:
                mlflow.log_param("params",None)
                
            mlflow.log_param("training_time",duration)
            
            preds = h2o_gbm.predict(self.test_set).as_data_frame()['p1']
            y_true = self.test_set['reordered'].as_data_frame()['reordered']

            # log evaluation metrics , dataset path,script path, environment
            self.__log_details(y_true,preds,prev_commit_hash,params) 

        del y_true,preds
        return h2o_gbm
    
    def train_xgb_rf(self,prev_commit_hash,params=None):
        
        dtrain = xgb.DMatrix(self.train_set.drop("reordered",axis=1), label = self.train_set['reordered'])
        dtest = xgb.DMatrix(self.test_set.drop("reordered",axis=1), label = self.test_set['reordered'])
        
        if params != None:
            
            if 'booster' not in params.keys():
                params['booster'] = 'gbtree'
                
            start = time.time()
            xgb_rf = xgb.train(params = params,dtrain=dtrain,num_boost_round=1)

        else :
            params = {
                'booster':'gbtree',
                'objective':'binary:logistic'
            }
            
            start = time.time()
            xgb_rf = xgb.train(params = params,dtrain=dtrain,num_boost_round=1)
            
        duration = get_time(start)

        with mlflow.start_run():

            mlflow.xgboost.log_model(xgb_rf,"xgb_rf_model")
            
            mlflow.log_param("training_time",duration)

            preds = xgb_rf.predict(dtest)
            y_true = self.test_set['reordered']

            self.__log_details(y_true,preds,prev_commit_hash,params)
            
        del y_true,preds,dtrain,dtest
        return xgb_rf

    def train_xgb_gbm(self,prev_commit_hash,params=None):
        
        dtrain = xgb.DMatrix(self.train_set.drop("reordered",axis=1), label = self.train_set['reordered'])
        dtest = xgb.DMatrix(self.test_set.drop("reordered",axis=1), label = self.test_set['reordered'])
        
        if params != None:
            
            if 'booster' not in params.keys():
                params['booster'] = 'gbtree'
                
            start = time.time()
            xgb_gbm = xgb.train(params = params,dtrain=dtrain,num_boost_round=1)

        else :
            params = {
                'booster':'gbtree',
                'objective':'binary:logistic'
            }
            
            start = time.time()
            xgb_gbm = xgb.train(params = params,dtrain=dtrain)
            
        duration = get_time(start)

        with mlflow.start_run():

            mlflow.xgboost.log_model(xgb_gbm,"xgb_gbm_model")
            
            mlflow.log_param("training_time",duration)

            preds = xgb_gbm.predict(dtest)
            y_true = self.test_set['reordered']

            self.__log_details(y_true,preds,prev_commit_hash,params)
            
        del y_true,preds,dtrain,dtest
        return xgb_gbm

    def train_lgb_gbm(self,prev_commit_hash,params=None):
        
        dtrain = lgb.Dataset(self.train_set.drop("reordered",axis=1),label=self.train_set['reordered'])
        dtest = lgb.Dataset(self.test_set.drop("reordered",axis=1),label=self.test_set['reordered'],reference = dtrain)

        if params != None:
            
            if  'verbose' not in params.keys():
                params['verbose'] = -1
                
            if 'objective'  not in params.keys():
                params['objective'] = 'binary'
                
            start = time.time()
            lgb_gbm = lgb.train(params,dtrain)

        else:
            
            params = {
                'verbose':-1,
                'objective':'binary'
            }
            start = time.time()
            lgb_gbm = lgb.train(params,dtrain)
            
        duration = get_time(start)

        with mlflow.start_run():

            mlflow.lightgbm.log_model(lgb_gbm,"lgb_gbm")
            
            mlflow.log_param("training_time",duration)
            preds = lgb_gbm.predict(self.test_set.drop("reordered",axis=1))
            y_true = self.test_set['reordered']
            self.__log_details(y_true,preds,prev_commit_hash,params)
            
        del y_true,preds,dtrain,dtest
        return lgb_gbm