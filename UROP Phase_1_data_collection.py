#!/usr/bin/env python
# coding: utf-8

# In[1]:


# UROP Phase 1 Data Retrieval and partial preprocessing
# all data is from material project database. 

from matminer.datasets import load_dataset
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
mpdr = MPDataRetrieval(api_key="KcDv6qi5w4rUZSlt")
d = load_dataset("heusler_magnetic")
heusler_formula = d['formula']


# In[ ]:


import pandas as pd
import time

query_time = list()
false_list = list()
true_list =list()
data = list()
heusler_matrix = pd.DataFrame()
start_time = time.time()
for name in heusler_formula:
    t_1 = time.time()
    data_got = mpdr.get_data(criteria = name,properties = ['pretty_formula','structure','elasticity'])
    t_2 = time.time()
    query_time.append(t_2 - t_1)
    data = data + data_got
data = pd.DataFrame(data)
end_time = time.time()
print('The program runs for {} with {} spent in query'.format(end_time - start_time, sum(query_time)))
end_time = time.time()


# In[5]:


data_2 = data.dropna(axis = 0)

# Rearranging the structure tensor columns
values_list = list()
new_cols_val = list()
tensor_list = list(data_2['elasticity'][1].keys())

for entry in data_2['elasticity']:
    values_list.append(list(entry.values()))

for prop in tensor_list:
    prop_value = list()
    for materials_val_list in values_list:
        prop_value.append(materials_val_list[tensor_list.index(prop)])
    new_cols_val.append(prop_value)

for prop_name in tensor_list:
    data_2[prop_name] = new_cols_val[tensor_list.index(prop_name)]

# prepare for featurization
from matminer.featurizers.conversions import StrToComposition
data_3 = StrToComposition().featurize_dataframe(data_2, "pretty_formula")
#data_3.columns


# In[9]:


# Saving this intermediate dataset before defining training data and targets
import numpy as np
np.savez_compressed("heusler_all.npz",data = data_3)


# In[ ]:


# Featurization
# This part is done with reference to the matiner examples
from matminer.featurizers.composition import ElementProperty

ep_feat = ElementProperty.from_preset(preset_name="magpie")
data_3 = ep_feat.featurize_dataframe(data_3, col_id="composition")

from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.composition import OxidationStates

data_3 = CompositionToOxidComposition().featurize_dataframe(data_3, "composition")

os_feat = OxidationStates()
data_3 = os_feat.featurize_dataframe(data_3, "composition_oxid")

from matminer.featurizers.structure import DensityFeatures

df_feat = DensityFeatures()
data_3 = df_feat.featurize_dataframe(data_3, "structure") 

unwanted_columns = ["elasticity","material_id","nsites", "compliance_tensor", "elastic_tensor", 
                    "elastic_tensor_original", "K_Voigt", "G_Voigt", "K_Reuss", "G_Reuss","warnings"]
data_4 = data_3.drop(unwanted_columns, axis = 1)


# In[ ]:


# Additional data cleaning after some trial runs
y = data_4['K_VRH'].values
excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "pretty_formula", 
            "poisson_ratio", "structure", "composition", "composition_oxid",
            "G_Voigt_Reuss_Hill","K_Voigt_Reuss_Hill","homogeneous_poisson","universal_anisotropy"]

X = data_4.drop(excluded, axis=1)

# The final row of excluded lable list is to minimize training interference. 
# An exceptionally good or near perfect linear fit was first obtained without dropping them, which is abnormal.
# The Random Forest model shows similiar syptoms.
# Importances list from Random forest reveals that these values,
# especially the G_Voigt_Reuss_Hill, take dominant role in predicting bulk modulus, rendering the other features as noise. 
# Therefore they are considered highly similiar quantities in nature with bulk modulus and dropped. 


# In[47]:


# Saving all the data for next phase
np.savez_compressed("heusler_mechanical.npz",
                    data = data_4,
                   column_label = data_4.columns.values,
                    X_column_label = X.columns.values,
                   X = X,
                   y = y)

