from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn import metrics
import matplotlib as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def outlier_removal_imputation(column_type, vitals_valid_range):
    column_range = vitals_valid_range[column_type]
    def outlier_removal_imputation_single_value(x):
        if x < column_range['outlier_low'] or x > column_range['outlier_high']:
            # set as missing
            return np.nan
        elif x < column_range['valid_low']:
            # impute with nearest valid value
            return column_range['valid_low']
        elif x > column_range['valid_high']:
            # impute with nearest valid value
            return column_range['valid_high']
        else:
            return x
    return outlier_removal_imputation_single_value

def convert_temp_to_celcius(df_master):
    for column in df_master.columns:
        column_type = column.split('_')[1] if len(column.split('_')) > 1 else None
        if column_type == 'temperature':
            # convert to celcius
            df_master[column] -= 32
            df_master[column] *= 5/9
    return df_master

def read_vitalsign_table(vitalsign_table_path):
    df_vitalsign = pd.read_csv(vitalsign_table_path, compression='gzip')
    vital_rename_dict = {vital: '_'.join(['ed', vital]) for vital in
                         ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'rhythm', 'pain']}
    df_vitalsign.rename(vital_rename_dict, axis=1, inplace=True)

    df_vitalsign['ed_pain'] = df_vitalsign['ed_pain'].apply(convert_str_to_float).astype(float)
    return df_vitalsign

def remove_outliers(df_master, vitals_valid_range):
    for column in df_master.columns:
        column_type = column.split('_')[1] if len(column.split('_')) > 1 else None
        if column_type in vitals_valid_range:
            df_master[column] = df_master[column].apply(outlier_removal_imputation(column_type, vitals_valid_range))
    return df_master

def preprocessing(path,df_train,df_test):
    print('Before filtering: training size =', len(df_train), ', testing size =', len(df_test))
    df_train = df_train[(df_train['outcome_hospitalization'] == False)]
    df_test = df_test[(df_test['outcome_hospitalization'] == False)].reset_index()
    print('After filtering: training size =', len(df_train), ', testing size =', len(df_test))
    outcome = "outcome_ed_revisit_3d"

    ed_variable = ["ed_temperature_last", "ed_heartrate_last", "ed_resprate_last", 
                "ed_o2sat_last", "ed_sbp_last", "ed_dbp_last", "ed_los", "n_med", "n_medrecon"]
    #ed_triage table
    eci_variable = ["eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2",  
                "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
                "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss", 
                "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression"] 
    #An Elixhauser comorbidity score of >15 can be used as a cut-off value with a 1-year mortality of 38%. 
    # This cut-off value is based on the ROC curve and the clinical interpretation of a reasonable life expectancy. The score of >15 is the highest tertile.
    # The high-risk patients with a score >15 should be considered carefully in the heart team, also with regards to other valve lesions and the clippability of the anatomy of the mitral valve. 
    # The prognosis should also be discussed with the patient in order to achieve shared decision-making.

    cci_variable = ["cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", "cci_Pulmonary", 
                "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", "cci_DM2", 
                "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", "cci_Cancer2", 
                "cci_HIV"] # Based on the CCI score, the severity of comorbidity was categorized into three grades: mild, with CCI scores of 1-2; moderate, with CCI scores of 3-4; and severe, with CCI scores ≥5.

    chiefcom_variable = ["chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache", 
                "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
                "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
                "chiefcom_dizziness"] #a concise statement describing the symptom, problem, condition, diagnosis, physician-recommended return, or other reason(true or false)

    patient_variable = ["age", "gender"] #general information

    n_variable = ["n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
                "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d"] #past_visit

    triage_variable = ["triage_pain", "triage_acuity"]

    variable = ["age", "gender", 
                
                "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
                "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", 
                
                "triage_pain", "triage_acuity",
                
                "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache", 
                "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
                "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
                "chiefcom_dizziness",
                
                "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", "cci_Pulmonary", 
                "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", "cci_DM2", 
                "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", "cci_Cancer2", 
                "cci_HIV",
                
                "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2",  
                "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
                "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss", 
                "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression",
                
                "ed_temperature_last", "ed_heartrate_last", "ed_resprate_last", 
                "ed_o2sat_last", "ed_sbp_last", "ed_dbp_last", "ed_los", "n_med", "n_medrecon"]


    X_train = df_train[variable].copy()
    ed_train = df_train[ed_variable].copy()
    eci_train = df_train[eci_variable].copy()
    cci_train = df_train[cci_variable].copy()
    chiefcom_train = df_train[chiefcom_variable].copy()
    patient_train = df_train[patient_variable].copy()
    n_train = df_train[n_variable].copy()
    triage_train = df_train[triage_variable].copy()
    y_train = df_train[outcome].copy()

    X_test = df_test[variable].copy()
    ed_test = df_test[ed_variable].copy()
    eci_test  = df_test[eci_variable].copy()
    cci_test  = df_test[cci_variable].copy()
    chiefcom_test  = df_test[chiefcom_variable].copy()
    patient_test  = df_test[patient_variable].copy()
    n_test  = df_test[n_variable].copy()
    triage_test  = df_test[triage_variable].copy()
    y_test = df_test[outcome].copy()

    resample_freq = '1H' #'30T'
    df_vitalsign = pd.read_csv(os.path.join(path, 'ed_vitalsign_' + resample_freq + '_resampled.csv'))
    encoder = LabelEncoder()
    X_train['gender'] = encoder.fit_transform(X_train['gender'])
    X_test['gender'] = encoder.transform(X_test['gender'])
    X_train['ed_los'] = pd.to_timedelta(X_train['ed_los']).dt.seconds / 60
    X_test['ed_los'] = pd.to_timedelta(X_test['ed_los']).dt.seconds / 60
    
    return X_train,y_train,X_test,y_test

def LR_result(X_train,y_train,X_test,y_test,confidence_interval,random_seed):
    lo=LogisticRegression(random_state=1)
    start = time.time()
    lo.fit(X_train,y_train)
    runtime = time.time()-start
    probs = lo.predict_proba(X_test)
    result = PlotROCCurve(probs[:,1],y_test, ci=confidence_interval, random_seed=random_seed)
    return lo,result,runtime

def add_score_CCI(df):
    conditions = [
        (df['age'] < 50),
        (df['age'] >= 50) & (df['age'] <= 59),
        (df['age'] >= 60) & (df['age'] <= 69),
        (df['age'] >= 70) & (df['age'] <= 79),
        (df['age'] >= 80)
    ]
    values = [0, 1, 2, 3, 4]
    df['score_CCI'] = np.select(conditions, values)    
    df['score_CCI'] = df['score_CCI'] + df['cci_MI'] + df['cci_CHF'] + df['cci_PVD'] + df['cci_Stroke'] + df['cci_Dementia'] + df['cci_Pulmonary'] + df['cci_PUD'] + df['cci_Rheumatic'] +df['cci_Liver1']*1 + df['cci_Liver2']*3 + df['cci_DM1'] + df['cci_DM2']*2 +df['cci_Paralysis']*2 + df['cci_Renal']*2 + df['cci_Cancer1']*2 + df['cci_Cancer2']*6 + df['cci_HIV']*6
    print("Variable 'add_score_CCI' successfully added")

def add_triage_MAP(df):
    df['triage_MAP'] = df['triage_sbp']*1/3 + df['triage_dbp']*2/3
    print("Variable 'add_triage_MAP' successfully added")

def add_score_REMS(df):
    conditions1 = [
        (df['age'] < 45),
        (df['age'] >= 45) & (df['age'] <= 54),
        (df['age'] >= 55) & (df['age'] <= 64),
        (df['age'] >= 65) & (df['age'] <= 74),
        (df['age'] > 74)
    ]
    values1 = [0, 2, 3, 5, 6]
    conditions2 = [
        (df['triage_MAP'] > 159),
        (df['triage_MAP'] >= 130) & (df['triage_MAP'] <= 159),
        (df['triage_MAP'] >= 110) & (df['triage_MAP'] <= 129),
        (df['triage_MAP'] >= 70) & (df['triage_MAP'] <= 109),
        (df['triage_MAP'] >= 50) & (df['triage_MAP'] <= 69),
        (df['triage_MAP'] < 49)
    ]
    values2 = [4, 3, 2, 0, 2, 4]
    conditions3 = [
        (df['triage_heartrate'] >179),
        (df['triage_heartrate'] >= 140) & (df['triage_heartrate'] <= 179),
        (df['triage_heartrate'] >= 110) & (df['triage_heartrate'] <= 139),
        (df['triage_heartrate'] >= 70) & (df['triage_heartrate'] <= 109),
        (df['triage_heartrate'] >= 55) & (df['triage_heartrate'] <= 69),
        (df['triage_heartrate'] >= 40) & (df['triage_heartrate'] <= 54),
        (df['triage_heartrate'] < 40)
    ]
    values3 = [4, 3, 2, 0, 2, 3, 4]
    conditions4 = [
        (df['triage_resprate'] > 49),
        (df['triage_resprate'] >= 35) & (df['triage_resprate'] <= 49),
        (df['triage_resprate'] >= 25) & (df['triage_resprate'] <= 34),
        (df['triage_resprate'] >= 12) & (df['triage_resprate'] <= 24),
        (df['triage_resprate'] >= 10) & (df['triage_resprate'] <= 11),
        (df['triage_resprate'] >= 6) & (df['triage_resprate'] <= 9),
        (df['triage_resprate'] < 6)
    ]
    values4 = [4, 3, 1, 0, 1, 2, 4]
    conditions5 = [
        (df['triage_o2sat'] < 75),
        (df['triage_o2sat'] >= 75) & (df['triage_o2sat'] <= 85),
        (df['triage_o2sat'] >= 86) & (df['triage_o2sat'] <= 89),
        (df['triage_o2sat'] > 89)
    ]
    values5 = [4, 3, 1, 0]
    df['score_REMS'] = np.select(conditions1, values1) + np.select(conditions2, values2) + np.select(conditions3, values3) +                              np.select(conditions4, values4) + np.select(conditions5, values5)
    print("Variable 'Score_REMS' successfully added")
    
def add_score_CART(df):
    conditions1 = [
        (df['age'] < 55),
        (df['age'] >= 55) & (df['age'] <= 69),
        (df['age'] >= 70) 
    ]
    values1 = [0, 4, 9]
    conditions2 = [
        (df['triage_resprate'] < 21),
        (df['triage_resprate'] >= 21) & (df['triage_resprate'] <= 23),
        (df['triage_resprate'] >= 24) & (df['triage_resprate'] <= 25),
        (df['triage_resprate'] >= 26) & (df['triage_resprate'] <= 29),
        (df['triage_resprate'] >= 30) 
    ]
    values2 = [0, 8, 12, 15, 22]
    conditions3 = [
        (df['triage_heartrate'] < 110),
        (df['triage_heartrate'] >= 110) & (df['triage_heartrate'] <= 139),
        (df['triage_heartrate'] >= 140) 
    ]
    values3 = [0, 4, 13]
    conditions4 = [
        (df['triage_dbp'] > 49),
        (df['triage_dbp'] >= 40) & (df['triage_dbp'] <= 49),
        (df['triage_dbp'] >= 35) & (df['triage_dbp'] <= 39),
        (df['triage_dbp'] < 35) 
    ]
    values4 = [0, 4, 6, 13]
    df['score_CART'] = np.select(conditions1, values1) + np.select(conditions2, values2) + np.select(conditions3, values3) +                              np.select(conditions4, values4)
    print("Variable 'Score_CART' successfully added")
    
def add_score_NEWS(df):
    conditions1 = [
        (df['triage_resprate'] <= 8),
        (df['triage_resprate'] >= 9) & (df['triage_resprate'] <= 11),
        (df['triage_resprate'] >= 12) & (df['triage_resprate'] <= 20),
        (df['triage_resprate'] >= 21) & (df['triage_resprate'] <= 24),
        (df['triage_resprate'] >= 25) 
    ]
    values1 = [3, 1, 0, 2, 3]
    conditions2 = [
        (df['triage_o2sat'] <= 91),
        (df['triage_o2sat'] >= 92) & (df['triage_o2sat'] <= 93),
        (df['triage_o2sat'] >= 94) & (df['triage_o2sat'] <= 95),
        (df['triage_o2sat'] >= 96) 
    ]
    values2 = [3, 2, 1, 0]
    conditions3 = [
        (df['triage_temperature'] <= 35),
        (df['triage_temperature'] > 35) & (df['triage_temperature'] <= 36),
        (df['triage_temperature'] > 36) & (df['triage_temperature'] <= 38),
        (df['triage_temperature'] > 38) & (df['triage_temperature'] <= 39),
        (df['triage_temperature'] > 39) 
    ]
    values3 = [3, 1, 0, 1, 2]
    conditions4 = [
        (df['triage_sbp'] <= 90),
        (df['triage_sbp'] >= 91) & (df['triage_sbp'] <= 100),
        (df['triage_sbp'] >= 101) & (df['triage_sbp'] <= 110),
        (df['triage_sbp'] >= 111) & (df['triage_sbp'] <= 219),
        (df['triage_sbp'] > 219) 
    ]
    values4 = [3, 2, 1, 0, 3]
    conditions5 = [
        (df['triage_heartrate'] <= 40),
        (df['triage_heartrate'] >= 41) & (df['triage_heartrate'] <= 50),
        (df['triage_heartrate'] >= 51) & (df['triage_heartrate'] <= 90),
        (df['triage_heartrate'] >= 91) & (df['triage_heartrate'] <= 110),
        (df['triage_heartrate'] >= 111) & (df['triage_heartrate'] <= 130),
        (df['triage_heartrate'] > 130) 
    ]
    values5 = [3, 1, 0, 1, 2, 3]    
    df['score_NEWS'] = np.select(conditions1, values1) + np.select(conditions2, values2) + np.select(conditions3, values3) +                              np.select(conditions4, values4) + np.select(conditions5, values5)
    print("Variable 'Score_NEWS' successfully added")
    
def add_score_NEWS2(df):   
    conditions1 = [
        (df['triage_resprate'] <= 8),
        (df['triage_resprate'] >= 9) & (df['triage_resprate'] <= 11),
        (df['triage_resprate'] >= 12) & (df['triage_resprate'] <= 20),
        (df['triage_resprate'] >= 21) & (df['triage_resprate'] <= 24),
        (df['triage_resprate'] >= 25) 
    ]
    values1 = [3, 1, 0, 2, 3]
    conditions2 = [
        (df['triage_temperature'] <= 35),
        (df['triage_temperature'] > 35) & (df['triage_temperature'] <= 36),
        (df['triage_temperature'] > 36) & (df['triage_temperature'] <= 38),
        (df['triage_temperature'] > 38) & (df['triage_temperature'] <= 39),
        (df['triage_temperature'] > 39) 
    ]
    values2 = [3, 1, 0, 1, 2]
    conditions3 = [
        (df['triage_sbp'] <= 90),
        (df['triage_sbp'] >= 91) & (df['triage_sbp'] <= 100),
        (df['triage_sbp'] >= 101) & (df['triage_sbp'] <= 110),
        (df['triage_sbp'] >= 111) & (df['triage_sbp'] <= 219),
        (df['triage_sbp'] > 219) 
    ]
    values3 = [3, 2, 1, 0, 3]
    conditions4 = [
        (df['triage_heartrate'] <= 40),
        (df['triage_heartrate'] >= 41) & (df['triage_heartrate'] <= 50),
        (df['triage_heartrate'] >= 51) & (df['triage_heartrate'] <= 90),
        (df['triage_heartrate'] >= 91) & (df['triage_heartrate'] <= 110),
        (df['triage_heartrate'] >= 111) & (df['triage_heartrate'] <= 130),
        (df['triage_heartrate'] > 130) 
    ]
    values4 = [3, 1, 0, 1, 2, 3]   
    df['score_NEWS2'] = np.select(conditions1, values1) + np.select(conditions2, values2) + np.select(conditions3, values3) +                              np.select(conditions4, values4)
    print("Variable 'Score_NEWS2' successfully added")
    
def add_score_MEWS(df):     
    conditions1 = [
        (df['triage_sbp'] <= 70),
        (df['triage_sbp'] >= 71) & (df['triage_sbp'] <= 80),
        (df['triage_sbp'] >= 81) & (df['triage_sbp'] <= 100),
        (df['triage_sbp'] >= 101) & (df['triage_sbp'] <= 199),
        (df['triage_sbp'] > 199) 
    ]
    values1 = [3, 2, 1, 0, 2]
    conditions2 = [
        (df['triage_heartrate'] <= 40),
        (df['triage_heartrate'] >= 41) & (df['triage_heartrate'] <= 50),
        (df['triage_heartrate'] >= 51) & (df['triage_heartrate'] <= 100),
        (df['triage_heartrate'] >= 101) & (df['triage_heartrate'] <= 110),
        (df['triage_heartrate'] >= 111) & (df['triage_heartrate'] <= 129),
        (df['triage_heartrate'] >= 130) 
    ]
    values2 = [2, 1, 0, 1, 2, 3]
    conditions3 = [
        (df['triage_resprate'] < 9),
        (df['triage_resprate'] >= 9) & (df['triage_resprate'] <= 14),
        (df['triage_resprate'] >= 15) & (df['triage_resprate'] <= 20),
        (df['triage_resprate'] >= 21) & (df['triage_resprate'] <= 29),
        (df['triage_resprate'] >= 30) 
    ]
    values3 = [2, 0, 1, 2, 3]
    conditions4 = [
        (df['triage_temperature'] < 35),
        (df['triage_temperature'] >= 35) & (df['triage_temperature'] < 38.5),
        (df['triage_temperature'] >= 38.5) 
    ]
    values4 = [2, 0, 2]        
    df['score_MEWS'] = np.select(conditions1, values1) + np.select(conditions2, values2) + np.select(conditions3, values3) +                              np.select(conditions4, values4) 
    print("Variable 'Score_MEWS' successfully added")
    
def add_score_SERP2d(df): 
    conditions1 = [
        (df['age'] < 30),
        (df['age'] >= 30) & (df['age'] <= 49),
        (df['age'] >= 50) & (df['age'] <= 79),
        (df['age'] >= 80)
    ]
    values1 = [0, 9, 13, 17]
    conditions2 = [
        (df['triage_heartrate'] < 60),
        (df['triage_heartrate'] >= 60) & (df['triage_heartrate'] <= 69),
        (df['triage_heartrate'] >= 70) & (df['triage_heartrate'] <= 94),
        (df['triage_heartrate'] >= 95) & (df['triage_heartrate'] <= 109),
        (df['triage_heartrate'] >= 110) 
    ]
    values2 = [3, 0, 3, 6, 10]
    conditions3 = [
        (df['triage_resprate'] < 16),
        (df['triage_resprate'] >= 16) & (df['triage_resprate'] <= 19),
        (df['triage_resprate'] >= 20) 
    ]
    values3 = [11, 0, 7]
    conditions4 = [
        (df['triage_sbp'] < 100),
        (df['triage_sbp'] >= 100) & (df['triage_sbp'] <= 114),
        (df['triage_sbp'] >= 115) & (df['triage_sbp'] <= 149),
        (df['triage_sbp'] >= 150) 
    ]
    values4 = [10, 4, 1, 0]
    conditions5 = [
        (df['triage_dbp'] < 50),
        (df['triage_dbp'] >= 50) & (df['triage_dbp'] <= 94),
        (df['triage_dbp'] >= 95) 
    ]
    values5 = [5, 0, 1]
    conditions6 = [
        (df['triage_o2sat'] < 90),
        (df['triage_o2sat'] >= 90) & (df['triage_o2sat'] <= 94),
        (df['triage_o2sat'] >= 95) 
    ]
    values6 = [7, 5, 0]
    df['score_SERP2d'] = np.select(conditions1, values1) + np.select(conditions2, values2) + np.select(conditions3, values3) +                              np.select(conditions4, values4) + np.select(conditions5, values5) + np.select(conditions6, values6)
    print("Variable 'Score_SERP2d' successfully added")

def add_score_SERP7d(df): 
    conditions1 = [
        (df['age'] < 30),
        (df['age'] >= 30) & (df['age'] <= 49),
        (df['age'] >= 50) & (df['age'] <= 79),
        (df['age'] >= 80)
    ]
    values1 = [0, 10, 17, 21]
    conditions2 = [
        (df['triage_heartrate'] < 60),
        (df['triage_heartrate'] >= 60) & (df['triage_heartrate'] <= 69),
        (df['triage_heartrate'] >= 70) & (df['triage_heartrate'] <= 94),
        (df['triage_heartrate'] >= 95) & (df['triage_heartrate'] <= 109),
        (df['triage_heartrate'] >= 110) 
    ]
    values2 = [2, 0, 4, 8, 12]
    conditions3 = [
        (df['triage_resprate'] < 16),
        (df['triage_resprate'] >= 16) & (df['triage_resprate'] <= 19),
        (df['triage_resprate'] >= 20) 
    ]
    values3 = [10, 0, 6]
    conditions4 = [
        (df['triage_sbp'] < 100),
        (df['triage_sbp'] >= 100) & (df['triage_sbp'] <= 114),
        (df['triage_sbp'] >= 115) & (df['triage_sbp'] <= 149),
        (df['triage_sbp'] >= 150) 
    ]
    values4 = [12, 6, 1, 0]
    conditions5 = [
        (df['triage_dbp'] < 50),
        (df['triage_dbp'] >= 50) & (df['triage_dbp'] <= 94),
        (df['triage_dbp'] >= 95) 
    ]
    values5 = [4, 0, 2]
    df['score_SERP7d'] = np.select(conditions1, values1) + np.select(conditions2, values2) + np.select(conditions3, values3) +                              np.select(conditions4, values4) + np.select(conditions5, values5)
    print("Variable 'Score_SERP7d' successfully added")
    
def add_score_SERP30d(df): 
    conditions1 = [
        (df['age'] < 30),
        (df['age'] >= 30) & (df['age'] <= 49),
        (df['age'] >= 50) & (df['age'] <= 79),
        (df['age'] >= 80)
    ]
    values1 = [0, 8, 14, 19]
    conditions2 = [
        (df['triage_heartrate'] < 60),
        (df['triage_heartrate'] >= 60) & (df['triage_heartrate'] <= 69),
        (df['triage_heartrate'] >= 70) & (df['triage_heartrate'] <= 94),
        (df['triage_heartrate'] >= 95) & (df['triage_heartrate'] <= 109),
        (df['triage_heartrate'] >= 110) 
    ]
    values2 = [1, 0, 2, 6, 9]
    conditions3 = [
        (df['triage_resprate'] < 16),
        (df['triage_resprate'] >= 16) & (df['triage_resprate'] <= 19),
        (df['triage_resprate'] >= 20) 
    ]
    values3 = [8, 0, 6]
    conditions4 = [
        (df['triage_sbp'] < 100),
        (df['triage_sbp'] >= 100) & (df['triage_sbp'] <= 114),
        (df['triage_sbp'] >= 115) & (df['triage_sbp'] <= 149),
        (df['triage_sbp'] >= 150) 
    ]
    values4 = [8, 5, 2, 0]
    conditions5 = [
        (df['triage_dbp'] < 50),
        (df['triage_dbp'] >= 50) & (df['triage_dbp'] <= 94),
        (df['triage_dbp'] >= 95) 
    ]
    values5 = [3, 0, 2]
    df['score_SERP30d'] = np.select(conditions1, values1) + np.select(conditions2, values2) + np.select(conditions3, values3) +                              np.select(conditions4, values4) + np.select(conditions5, values5) + df['cci_Cancer1']*6 + df['cci_Cancer2']*12
    print("Variable 'Score_SERP30d' successfully added")
    

def auc_with_ci(probs,y_test_roc, lower = 2.5, upper = 97.5, n_bootstraps=200, rng_seed=10):
    print(lower, upper)
    y_test_roc = np.asarray(y_test_roc)
    bootstrapped_auroc = []
    bootstrapped_ap = []
    bootstrapped_sensitivity = []
    bootstrapped_specificity = []

    rng = np.random.default_rng(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.integers(0, len(y_test_roc)-1, len(y_test_roc))
        if len(np.unique(y_test_roc[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        fpr, tpr, threshold = metrics.roc_curve(y_test_roc[indices],probs[indices])
        auroc = metrics.auc(fpr, tpr)
        ap = metrics.average_precision_score(y_test_roc[indices], probs[indices])
        a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
        sensitivity = tpr[a]
        specificity = 1-fpr[a]
        bootstrapped_auroc.append(auroc)
        bootstrapped_ap.append(ap)
        bootstrapped_sensitivity.append(sensitivity)
        bootstrapped_specificity.append(specificity)

    lower_auroc,upper_auroc = np.percentile(bootstrapped_auroc, [lower, upper])
    lower_ap,upper_ap = np.percentile(bootstrapped_ap, [lower, upper])
    lower_sensitivity,upper_sensitivity = np.percentile(bootstrapped_sensitivity, [lower, upper])
    lower_specificity,upper_specificity = np.percentile(bootstrapped_specificity, [lower, upper])

    std_auroc = np.std(bootstrapped_auroc)
    std_ap = np.std(bootstrapped_ap)
    std_sensitivity = np.std(bootstrapped_sensitivity)
    std_specificity = np.std(bootstrapped_specificity)

    return lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity

def CM_plot_train_test(model,X_train,y_train,X_test,y_test):
    new_probs_train = model.predict(X_train)
    # cm1 = metrics.confusion_matrix(y_train,new_probs_train)
    # train_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm1, display_labels = [False, True])
    # train_cm_display.plot()
    new_probs_test = model.predict(X_test)
    # cm1 = metrics.confusion_matrix(y_test,new_probs_test)
    # train_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm1, display_labels = [False, True])
    # train_cm_display.plot()
    fig, ax = plt.pyplot.subplots(1, 2,figsize=(15,10))
    ax[0].set_title("test")
    ax[1].set_title("train")

    metrics.ConfusionMatrixDisplay(
        confusion_matrix=metrics.confusion_matrix(y_test, new_probs_test), 
        display_labels=[False, True]).plot(ax=ax[0])

    metrics.ConfusionMatrixDisplay(
        confusion_matrix=metrics.confusion_matrix(y_train, new_probs_train), 
        display_labels=[False, True]).plot(ax=ax[1])
    
    return

def tree_based_fs(X_train, y_train,X_test):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)
    clf.feature_importances_  
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X_train)
    Xtest_new = model.transform(X_test)
    feature_idx = model.get_support()
    feature_name = X_train.columns[feature_idx]
    return feature_name,X_new,Xtest_new

def varianceThreshold_fs(X_train, y_train,X_test):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_new = sel.fit_transform(X_train)
    Xtest_new = sel.fit_transform(X_test)
    sel.fit(X_train)
    feature_idx = sel.get_support()
    feature_name = X_train.columns[feature_idx]
    return feature_name,X_new,Xtest_new

def ROC_result(probs,y_test_roc, ci= 95, random_seed=0):
    fpr, tpr, threshold = metrics.roc_curve(y_test_roc,probs)
    roc_auc = metrics.auc(fpr, tpr)
    average_precision = average_precision_score(y_test_roc, probs)
    a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
    sensitivity = tpr[a]
    specificity = 1-fpr[a]
    threshold = threshold[a]
    lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity = auc_with_ci(probs,y_test_roc, lower = (100-ci)/2, upper = 100-(100-ci)/2, n_bootstraps=20, rng_seed=random_seed)
    precision, recall, threshold2 = precision_recall_curve(y_test_roc, probs)
    return [roc_auc, average_precision, sensitivity, specificity, threshold, lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity]
    
def CM_plot_train_test_DL(model,X_train,y_train,X_test,y_test):
    
    probs_test = model.predict(X_test)
    probs_train = model.predict(X_train)

    fig, ax = plt.pyplot.subplots(1, 2,figsize=(15,10))
    ax[0].set_title("test")
    ax[1].set_title("train")
    result_test = ROC_result(probs_test, y_test, ci=95, random_seed=0)
    result_train = ROC_result(probs_train, y_train, ci=95, random_seed=0)
    new_probs_test = []
    new_probs_train = []

    for x in probs_test:
        if x[0] < float(result_test[4]):
            new_probs_test.append(False)
        else:
            new_probs_test.append(True)

    for x in probs_train:
        if x[0] < float(result_train[4]):
            new_probs_train.append(False)
        else:
            new_probs_train.append(True)
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=metrics.confusion_matrix(y_test, new_probs_test), 
        display_labels=[False, True]).plot(ax=ax[0],values_format="d")

    metrics.ConfusionMatrixDisplay(
        confusion_matrix=metrics.confusion_matrix(y_train, new_probs_train), 
        display_labels=[False, True]).plot(ax=ax[1],values_format="d")
    
    return

def convert_data(x_train,x_test,variable,df_vitalsign,ed = False,patient = False):

    variable_with_id = ["stay_id"]
    variable_with_id.extend(variable)
    x1_cols = [x for x in variable_with_id[1:] if not ('ed' in x and 'last' in x)]
    x2_cols = [x for x in df_vitalsign.columns if 'ed' in x]
    if ed == True:
        x_train['ed_los'] = pd.to_timedelta(x_train['ed_los']).dt.seconds / 60
        x_test['ed_los'] = pd.to_timedelta(x_test['ed_los']).dt.seconds / 60
    if patient == True:
        encoder = LabelEncoder()
        x_train['gender'] = encoder.fit_transform(x_train['gender'])
        x_test['gender'] = encoder.transform(x_test['gender'])

    x_train = x_train[x1_cols].to_numpy().astype(np.float64)
    x_train = np.array(pad_sequences(x_train))
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)

    x_test = x_test[x1_cols].to_numpy().astype(np.float64)
    x_test = np.array(pad_sequences(x_test))
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    return x_train,x_test


def convert_data1(x_train,x_test):
    X_train1 = x_train.to_numpy().astype(np.float64)
    X_train1 = np.array(pad_sequences(X_train1))
    X_train1 = X_train1.reshape(X_train1.shape[0],X_train1.shape[1],1)


    X_test1 = x_test.to_numpy().astype(np.float64)
    X_test1 = np.array(pad_sequences(X_test1))
    X_test1 = X_test1.reshape(X_test1.shape[0],X_test1.shape[1],1)
    return X_train1,X_test1
    
def PlotROCCurve_multilstm(path,probs,y_test_roc, ci= 95, random_seed=0):
    
    fpr, tpr, threshold = metrics.roc_curve(y_test_roc,probs)
    roc_auc = metrics.auc(fpr, tpr)
    average_precision = average_precision_score(y_test_roc, probs)
    a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
    sensitivity = tpr[a]
    specificity = 1-fpr[a]
    threshold = threshold[a]
    print("AUC:",roc_auc)
    print("AUPRC:", average_precision)
    print("Sensitivity:",sensitivity)
    print("Specificity:",specificity)
    print("Score thresold:",threshold)
    lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity = auc_with_ci(probs,y_test_roc, lower = (100-ci)/2, upper = 100-(100-ci)/2, n_bootstraps=20, rng_seed=random_seed)


    plt.title('Receiver Operating Characteristic: AUC={0:0.4f}'.format(
          roc_auc))
    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path+'/ROC_Curve.png')
    plt.show()

    precision, recall, threshold2 = precision_recall_curve(y_test_roc, probs)
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: AUPRC={0:0.4f}'.format(
          average_precision))
    plt.savefig(path+'/Precision-Recall_Curve.png')
    plt.show()
    return [roc_auc, average_precision, sensitivity, specificity, threshold, lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity]

def PlotROCCurve(probs,y_test_roc, ci= 95, random_seed=0):
    
    fpr, tpr, threshold = metrics.roc_curve(y_test_roc,probs)
    roc_auc = metrics.auc(fpr, tpr)
    average_precision = average_precision_score(y_test_roc, probs)
    a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
    sensitivity = tpr[a]
    specificity = 1-fpr[a]
    threshold = threshold[a]
    print("AUC:",roc_auc)
    print("AUPRC:", average_precision)
    print("Sensitivity:",sensitivity)
    print("Specificity:",specificity)
    print("Score thresold:",threshold)
    lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity = auc_with_ci(probs,y_test_roc, lower = (100-ci)/2, upper = 100-(100-ci)/2, n_bootstraps=20, rng_seed=random_seed)


    plt.title('Receiver Operating Characteristic: AUC={0:0.4f}'.format(
          roc_auc))
    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    precision, recall, threshold2 = precision_recall_curve(y_test_roc, probs)
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: AUPRC={0:0.4f}'.format(
          average_precision))
    plt.show()
    return [roc_auc, average_precision, sensitivity, specificity, threshold, lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity]

def preprocessing_full_feature(path,df_train,df_test):
    print('Before filtering: training size =', len(df_train), ', testing size =', len(df_test))
    df_train = df_train[(df_train['outcome_hospitalization'] == False)]
    df_test = df_test[(df_test['outcome_hospitalization'] == False)].reset_index()
    print('After filtering: training size =', len(df_train), ', testing size =', len(df_test))
    outcome = "outcome_ed_revisit_3d"

   
    X_train = df_train.drop("outcome_ed_revisit_3d", axis=1)
    X_train = df_train.drop("outcome_ed_revisit_3d", axis=1)
    X_train = df_train.drop("outcome_ed_revisit_3d", axis=1)
    y_train = df_train[outcome].copy()

    X_test = df_test.drop("outcome_ed_revisit_3d", axis=1)
    X_test = df_test.drop("outcome_ed_revisit_3d", axis=1)
    X_test = df_test.drop("outcome_ed_revisit_3d", axis=1)
   
    y_test = df_test[outcome].copy()

    resample_freq = '1H' #'30T'
    df_vitalsign = pd.read_csv(os.path.join(path, 'ed_vitalsign_' + resample_freq + '_resampled.csv'))
    encoder = LabelEncoder()
    X_train['gender'] = encoder.fit_transform(X_train['gender'])
    X_test['gender'] = encoder.transform(X_test['gender'])
    X_train['ed_los'] = pd.to_timedelta(X_train['ed_los']).dt.seconds / 60
    X_test['ed_los'] = pd.to_timedelta(X_test['ed_los']).dt.seconds / 60
    
    return X_train,y_train,X_test,y_test

def find_sens_speci(clf,X_test,y_test):
    yp = clf.predict(X_test)
    cm1 = confusion_matrix(y_test,yp)

    total1=sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1=(cm1[0,0]+cm1[1,1])/total1

    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    
    f = f1_score(y_test,yp)

    return f, accuracy1, sensitivity1, specificity1

