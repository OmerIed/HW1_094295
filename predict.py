import pandas as pd
import numpy as np
import sys
import os
import random 
import pickle
import xgboost
# Set the random seed for numpy
np.random.seed(42)
# Set the random seed for Python's built-in random module
random.seed(42)

if len(sys.argv) != 2:
    print("Usage: python script.py <path>")
    exit(1)

directory_path=sys.argv[1]

def drop_redundent(df):
  df.drop(['Unit1','Unit2','ICULOS'],axis=1, inplace=True)
  return df

def aggregation(df):
  freq = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
  rare = [ 'EtCO2','BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
            'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
            'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
            'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
            'Fibrinogen', 'Platelets']
  statistics = {}
  df_conc = pd.DataFrame()
  counter = 0
  for patient in df["patient"].unique():
    statistics[patient] = {}
    counter +=1
    if counter%1000 == 0:
      print(counter)
    pat_desc = df[df["patient"]==patient].describe()
    statistics[patient]['patient'] = patient
    patient_count = df[df["patient"]==patient].shape[0]
    statistics[patient]['count']=patient_count
    statistics[patient]['Age']=pat_desc["Age"][1]
    statistics[patient]['Gender']=pat_desc["Gender"][1]
    statistics[patient]['HospAdmTime'] = pat_desc["HospAdmTime"][1]
    for col in freq:
      stat_list = pat_desc[col].tolist()
      statistics[patient][f"mean {col}"] = stat_list[1]
      statistics[patient][f"Q1 {col}"] = stat_list[4]
      statistics[patient][f"Q2 {col}"] = stat_list[5]
      statistics[patient][f"Q3 {col}"] = stat_list[6]
      statistics[patient][f"max {col}"] = stat_list[7]
      statistics[patient][f"min {col}"] = stat_list[3]
      statistics[patient][f"std {col}"] = stat_list[2]
      statistics[patient][f"p1 {col}"] = df[df["patient"]==patient][col].iloc[:int(patient_count/4)].mean()
      statistics[patient][f"p2 {col}"] = df[df["patient"]==patient][col].iloc[int(patient_count/4):int(patient_count/2)].mean()
      statistics[patient][f"p3 {col}"] = df[df["patient"]==patient][col].iloc[int(patient_count/2):int((3*patient_count)/4)].mean()
      statistics[patient][f"p4 {col}"] = df[df["patient"]==patient][col].iloc[int((3*patient_count)/4):int(patient_count)].mean()
    for col in rare:
      stat_list = pat_desc[col].tolist()
      statistics[patient][f"max {col}"] = stat_list[7]
      statistics[patient][f"min {col}"] = stat_list[3]
    
  df_conc = pd.DataFrame.from_dict(statistics,orient="index")
  return df_conc

def categorial_imputation(df,rare):
  for col in rare:
    print(col)
    col_desc = df[col].describe()
    for patient in df["patient"]:
      # print(df[df["patient"]==patient][col])
      val = df[df["patient"]==patient][col].iloc[0]
      if val == np.nan:
        df.loc[df['patient'] == patient,col]=0
      elif val>=col_desc[3] and val<col_desc[4]:
        df.loc[df['patient'] == patient,col]= 1
      elif val>=col_desc[4] and val<col_desc[5]:
        df.loc[df['patient'] == patient,col]= 2
      elif val>=col_desc[5] and val<col_desc[6]:
        df.loc[df['patient'] == patient,col] = 3
      else:
        df.loc[df['patient'] == patient,col] = 4
  return df

def mean_imputation(df):
  col_mean = df.mean()
  df_imp = df.fillna(col_mean)
  return df_imp

def extract_to_df(directory_path):
  df_test = pd.DataFrame()
  for directory in [directory_path]:
    for filename in os.listdir(directory):
      if filename.endswith('.psv'): 
        # Load the data from the PSV file into a dataframe
        file_path = os.path.join(directory, filename)
        file_data = pd.read_csv(file_path, sep='|')
        file_data['patient'] = str(filename).split('.')[0].split('_')[1]
        index_up_to = len(file_data) - len(file_data[file_data['SepsisLabel'] == 1])
        file_data = file_data[file_data.index<=index_up_to]
        df_test = pd.concat([df_test, file_data])
  return df_test

df_test = extract_to_df(directory_path)
print('-----extracted to df-------')
df_test = drop_redundent(df_test)
df_test = aggregation(df_test)
print('------finished aggregation------')
miss_alot = [ 'max EtCO2','min EtCO2','max BaseExcess','min BaseExcess','max HCO3','min HCO3','max FiO2',
 'min FiO2','max pH','min pH','max PaCO2','min PaCO2','max SaO2','min SaO2','max AST','min AST',
 'max Alkalinephos','min Alkalinephos','max Chloride','min Chloride','max Bilirubin_direct',
 'min Bilirubin_direct','max Lactate','min Lactate','max Bilirubin_total','min Bilirubin_total',
 'max TroponinI','min TroponinI','max PTT','min PTT','max Fibrinogen', 'min Fibrinogen']
df_test = categorial_imputation(df_test,miss_alot)
df_test = mean_imputation(df_test) 
#df_test.to_csv('temp_for_failure.csv',index=False)

#df_test = pd.read_csv('temp_for_failure.csv')
patient_list = df_test['patient']
patient_list = [int(x) for x in patient_list]
X_test = df_test.drop(['patient'],axis=1).to_numpy()

file_name = "final_model.pkl"
# load
xgb_model = pickle.load(open(file_name, "rb"))
print('-----Loaded model, starting prediction------')
y_preds = xgb_model.predict_proba(X_test)
y_pred_binary = [1 if p[1] >= 0.34 else 0 for p in y_preds]
df_preds = pd.DataFrame({'id':patient_list, 'prediction':y_pred_binary})

df_preds['id'] = df_preds['id'].apply(lambda x: f'patient_{str(x)}')
df_preds['id_num'] = df_preds['id'].str.split('_').str[-1].astype(int)

# sort the dataframe by id_num column
df_sorted = df_preds.sort_values('id_num')

# drop the id_num column if not needed
df_sorted = df_sorted.drop('id_num', axis=1)
df_sorted.to_csv('prediction.csv', index=False)

