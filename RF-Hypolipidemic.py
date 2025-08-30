# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 17:44:23 2025

@author: 于皓
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, RocCurveDisplay
from sklearn.metrics import recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, precision_score
from modlamp.descriptors import GlobalDescriptor

# ----------- Raw data loading and feature extraction ------------
data = np.array(pd.read_csv(r'D:\ML2\na_data.csv'))
data = [i[0] for i in data]
desc = GlobalDescriptor(data)
desc.calculate_all()
a = desc.descriptor
five_features = np.concatenate((a[:, 2:5], a[:, 6:8]), axis=1)

pep_po = []
with open(r'C:\Users\CD\Desktop\Data-po.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        peptide = line.strip()
        if '>' not in peptide:
            pep_po.append(peptide)
des = GlobalDescriptor(pep_po)
des.calculate_all()
b = des.descriptor
five_features_po = np.concatenate((b[:, 2:5], b[:, 6:8]), axis=1)

X = np.concatenate((five_features, five_features_po), axis=0)
y = np.concatenate((np.zeros(five_features.shape[0]), np.ones(five_features_po.shape[0])))

# ----------- Data partitioning ------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------- Random Forest model training (using optimal parameters)------------
rf_raw = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5, criterion='entropy', random_state=42)
rf_raw.fit(X_train_scaled, y_train)

####gc_data
gc_data = np.array(pd.read_csv(r'C:\Users\CD\Desktop\toxic_data.csv'))
gc_data = [i[0] for i in gc_data]
desc = GlobalDescriptor(gc_data)
desc.calculate_all()
gc_features = desc.descriptor
five_gcfeatures = np.concatenate((gc_features[:, 2:5], gc_features[:, 6:8]), axis=1)
scale_five_gcfeatures = scaler.transform(five_gcfeatures)
result_pro = rf_raw.predict_proba(scale_five_gcfeatures)[:,1]
result = rf_raw.predict(scale_five_gcfeatures)
pep_names = np.array(gc_data).reshape((-1,1))
# print(result_pro.shape)
# pd_reasult = pd.DataFrame(np.concatenate((pep_names,five_gcfeatures,result_pro.reshape((-1,1)),result.reshape((-1,1))),axis=1)).to_csv(r'C:\Users\CD\Desktop\实验数据\pepML终极代码\筛选结果\Lipid-lowering peptide results.csv',index=False)
j = np.concatenate((pep_names,five_gcfeatures,result_pro.reshape((-1,1)),result.reshape((-1,1))),axis=1)
i_sort = np.argsort(j[:,-2])
jj = j[i_sort]
# pd_reasult = pd.DataFrame(jj).to_csv(r'C:\Users\CD\Desktop\Lipid-lowering peptide results.csv',index=False)
