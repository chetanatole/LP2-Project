import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


class AirQuality:
    
    dataset = ""
    x = ""
    y = ""
    x_train = "" 
    x_test = "" 
    y_train = "" 
    y_test = ""
    RandomForestModel = ""
    XgbModel = ""
    SvmModel = ""
    DecisionTreeModel = ""
    le_X_city = ""
    le_X_date = ""
    le_Y = ""

    
    def readCsv(self, file_name):
        
        #Importing the datset
        self.dataset = pd.read_csv(file_name)
        
        self.dataset.dropna(axis=0, subset = ["Air_quality", "Xylene", "AQI", "Toluene",
                                         "Benzene", "O3", "SO2", "CO", "NH3", "NOx", 
                                         "NO2", "PM10", "PM2.5", "NO"], how = 'all', inplace= True)
        self.dataset.dropna(subset = ["Air_quality"], inplace=True)
#         print("asasd")
             
        self.x = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:,15].values
        
        #Filling the missing values
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
        imputer = imputer.fit(self.x[:,2:15])
        self.x[:,2:15] = imputer.transform(self.x[:,2:15])
        
        
        #Encoding the attributes
        self.le_X_city = LabelEncoder()
        self.le_X_date = LabelEncoder()
        self.le_Y = LabelEncoder()
        self.y = self.le_Y.fit_transform(self.y)
        
        self.x[:,0] = self.le_X_city.fit_transform(self.x[:,0])
        self.x[:,1] = self.le_X_date.fit_transform(self.x[:,1])
    
    
        #Splitting the dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = 0.3, random_state = 0)
        
        
        #print('Classes and number of values in trainset',Counter(self.y_train))
        from imblearn.over_sampling import SMOTE
        oversample = SMOTE()
        self.x_train, self.y_train = oversample.fit_resample(self.x_train,self.y_train)
        #print('Classes and number of values in trainset after SMOTE:',Counter(self.y_train))
        self.med=np.median(self.x_train,axis=0)
    
    def trainRF(self):
        self.RandomForestModel=RandomForestClassifier(n_estimators=100,random_state = 0)
        self.RandomForestModel.fit(self.x_train,self.y_train)
    
    def trainXGB(self):
        self.XgbModel = XGBClassifier()
        self.XgbModel.fit(self.x_train, self.y_train)
    
    def trainSVM(self):
        self.SvmModel = SVC(kernel = "linear",random_state = 0)
        self.SvmModel.fit(self.x_train, self.y_train)
    
    def trainDT(self):
        self.DecisionTreeModel = DecisionTreeClassifier(random_state = 0)
        self.DecisionTreeModel.fit(self.x_train, self.y_train)
    
    def RandomForest(self):
        #RandomForstClassifier Model
        
        self.y_pred=self.RandomForestModel.predict(self.x_test)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        
        a = accuracy_score(self.y_test,self.y_pred)
        precision = precision_score(self.y_test,self.y_pred, average='micro')
        recall = recall_score(self.y_test,self.y_pred, average='micro')        
        f1 = f1_score(self.y_test,self.y_pred, average='micro')

        return cm, a*100, precision*100, recall*100, f1*100
    
    def XGB(self):
        #XGBCLassifier Model
        
        self.y_pred = self.XgbModel.predict(self.x_test)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        
        a = accuracy_score(self.y_test,self.y_pred)
        precision = precision_score(self.y_test,self.y_pred, average='micro')
        recall = recall_score(self.y_test,self.y_pred, average='micro')        
        f1 = f1_score(self.y_test,self.y_pred, average='micro')
        return cm, a*100, precision*100, recall*100, f1*100
    
    def SVC(self):
        #SVC Model

        self.y_pred = self.SvmModel.predict(self.x_test)
        
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        
        a = accuracy_score(self.y_test,self.y_pred)
        precision = precision_score(self.y_test,self.y_pred, average='micro')
        recall = recall_score(self.y_test,self.y_pred, average='micro')        
        f1 = f1_score(self.y_test,self.y_pred, average='micro')
        return cm, a*100, precision*100, recall*100, f1*100
    
    def DecisionTree(self):
        #DecisionTreeClassifier Model
      
        self.y_pred = self.DecisionTreeModel.predict(self.x_test)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        
        a = accuracy_score(self.y_test,self.y_pred)
        precision = precision_score(self.y_test,self.y_pred, average='micro')
        recall = recall_score(self.y_test,self.y_pred, average='micro')        
        f1 = f1_score(self.y_test,self.y_pred, average='micro')
        return cm, a*100, precision*100, recall*100, f1*100
    
    def predict(self,City,Date,PM25,PM10,NO,NO2,NOx,NH3,CO,SO2,O3,Benzene,Toluene,Xylene,AQI):
       
        res = []
        city = self.le_X_city.fit_transform([City])
        date = self.le_X_date.fit_transform([Date])
        
        if(not PM25):
            PM25val=self.med[2]
        else:
            PM25val=float(PM25)
        if(not PM10):
            PM10val=self.med[3]
        else:
            PM10val=float(PM10)
        if(not NO):
            NOval=self.med[4]
        else:
            NOval=float(NO)
        if(not NO2):
            NO2val=self.med[5]
        else:
            NO2val=float(NO2)
        if(not NOx):
            NOxval=self.med[6]
        else:
            NOxval=float(NOx)
        if(not NH3):
            NH3val=self.med[7]
        else:
            NH3val=float(NH3)
        if(not CO):
            COval=self.med[8]
        else:
            COval=float(CO)
        if(not SO2):
            SO2val=self.med[9]
        else:
            SO2val=float(SO2)
        if(not O3):
            O3val=self.med[10]
        else:
            O3val=float(O3)
        if(not Benzene):
            Benzeneval=self.med[11]
        else:
            Benzeneval=float(Benzene)
        if(not Toluene):
            Tolueneval=self.med[12]
        else:
            Tolueneval=float(Toluene)
        if(not Xylene):
            Xyleneval=self.med[13]
        else:
            Xyleneval=float(Xylene)
        if(not AQI):
            AQIval=self.med[14]
        else:
            AQIval=float(AQI)
        
        ls = [city[0] ,date[0], PM25val, PM10val, NOval, NO2val, NOxval, NH3val, COval, SO2val, O3val, Benzeneval, Tolueneval, Xyleneval, AQIval]
        lst = [];
        lst.append(ls);
        
        temp = self.le_Y.inverse_transform(self.RandomForestModel.predict(lst))
        temp = temp.tolist()
        res.append(temp[0])
        
        temp = self.le_Y.inverse_transform(self.SvmModel.predict(lst))
        temp = temp.tolist()
        res.append(temp[0])
              
        temp = self.le_Y.inverse_transform(self.DecisionTreeModel.predict(lst))
        temp = temp.tolist()
        res.append(temp[0]) 

#         print(lst)
        ll=np.array(lst).reshape(1,-1)
#         print(ll)
        temp = self.le_Y.inverse_transform(self.XgbModel.predict(ll))
        temp = temp.tolist()
        res.append(temp[0])

        
        return res        
        


    


  
