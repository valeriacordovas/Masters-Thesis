# -*- coding: utf-8 -*-
"""
Data Analysis using Machine-Learning - NSDUH 2018 and 2019

@author: Valeria Córdova Silveira
"""
import numpy as np
import pandas as pd
import time
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import train_test_split
from src.GUIDETree import GUIDETree
from models.OModel import OModel
import os, pickle
from stargazer.stargazer import Stargazer


""" Load data """
dta_file  = "G:\Mi unidad\Master\Thesis\Data\Final\Final_sample.dta"

ogdata = pd.read_stata(dta_file, convert_categoricals = False)


# ***********************************
#
#    Alcohol use
#
# ***********************************
""" Select variables and divide sample """
sample = ogdata[['alcuse_num', 'alcfage', 'smokeqnt', 'mjnfq', 'cocfq', 'age', 'sex', 'race', 'health', 'mhealth', 'married', 'emplmnt', 'educcat', 'fincome', 'socialsec', 'hhkids', 'relserv']].dropna(axis=0, how='any')

X = sample.iloc[:,1:]
header = list(X.columns)
header[11] = 'educ'
X = X.to_numpy()
y = sample[['alcuse_num']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
var_types = ['cont', 'ord', 'cont', 'cont', 'ord', 'bin', 'cat', 'ord', 'cont', 'cat', 'cat', 'ord', 'ord', 'bin', 'ord', 'ord']
classes = np.unique(y)

# Settings
save_model_tree = True  # save model tree?
save_model_tree_predictions = True  # save model tree predictions/explanations?
cross_validation = False  # cross-validate model tree?



# ===============================
#  Ordered Logit Model
# ===============================
### Train model tree
# Define model
model_ologit = OModel(OrderedModel, classes = classes, distr = 'logit')

# Build
tree_train = GUIDETree(model = model_ologit, max_depth=15, 
                      min_samples_leaf=400)

# Fit
print("Training model tree with '{}'...".format(model_ologit.__class__.__name__))
tic = time.time()
tree_train.fit(X_train, y_train, var_types = var_types, verbose=True)
toc = time.time()
print(f"Tree trained in {(toc-tic)/60:0.3f} minutes")


### Prune Tree
print("Pruning model tree")
tic = time.time()
pruning = tree_train.prune()
toc = time.time()
print(f"Pruning finished in {(toc-tic)/60:0.3f} minutes")

# Cross-validate pruned trees
test_loss  = []
for alpha in range(len(pruning)):
    test_loss.append([pruning[alpha][0], pruning[alpha][1].loss(X_test, y_test)])

test_loss = np.array(test_loss)
min_test_loss = np.min(test_loss[:,1])
best_alpha = test_loss[np.where(test_loss == min_test_loss)[0],0]

print(f" -> Best tree's loss = {min_test_loss:0.3f} with alpha = {best_alpha[0]:0.3f}")


final_tree = list(np.array(pruning)[np.where(pruning == best_alpha)[0],1])[0]



# ------------------------------- SAVE RESULTS  -------------------------------
# Export pruned tree as png
final_tree.export_graphviz(os.path.join("G:\Mi unidad\Master\Thesis\Document\Output\Figures", "final_400"), header,
                           export_png=True, export_pdf=False)


# Save full tree
file_to_store = open("Full_400.pickle", "wb")
pickle.dump(tree_train, file_to_store)
file_to_store.close()

# Save pruned tree
file_to_store = open("Pruned_400.pickle", "wb")
pickle.dump(final_tree, file_to_store)
file_to_store.close()


# Save alphas
file_to_store = open("alphas_400.pickle", "wb")
pickle.dump(pruning, file_to_store)
file_to_store.close()

# -----------------------------------------------------------------------------

""" Estimate Leaf Models """
# Get leaves information
leaves = final_tree.get_leaves()

## Estimate model on each leaf and calculate predicted probabilities for each category
obs_probas = np.empty((1,5))
pred_probas = np.empty((1,5))
leaf_model = {}

# Full-data model
y_full = pd.DataFrame(y_train, columns = ['alcuse'])
y_full[y_full == 0] = 'No use'
y_full[y_full == 1] = 'Moderate'
y_full[y_full == 2] = 'Misuse'
y_full[y_full == 3] = 'Binge'

y_full = np.squeeze(np.array(y_full),1)

y_full = pd.Categorical(y_full, categories=['No use', 'Moderate', 'Misuse', 'Binge'], ordered=True)
y_full = pd.Series(y_full)
    
cats_full = []
for i in range(np.size(X_train,1)):
    if var_types[i] == "cat" or var_types[i] == "ord":
        cats_full.append(header[i])    

Xest = pd.get_dummies(pd.DataFrame(X_train, columns = header), columns = cats_full, drop_first=True)

fit = model_ologit.fit(Xest, y_full, cov_type='HC0', method = 'bfgs', skip_hessian = False)
pred = model_ologit.predict(Xest)

leaf_model[1] = [fit]
obs_probas[0,0] = 0
obs_probas[0,1:] = np.reshape(np.array(y_full.value_counts(normalize=True) * 100), (1,4))
pred_probas[0,0] = 0
pred_probas[0,1:] = np.mean(pred, axis = 0)*100

# Leaves models
for i in range(len(leaves)):
    leaf = leaves[i]
    
    y_leaf = pd.DataFrame(leaf["endog"], columns = ['alcuse'])
    y_leaf[y_leaf == 0] = 'No use'
    y_leaf[y_leaf == 1] = 'Moderate'
    y_leaf[y_leaf == 2] = 'Misuse'
    y_leaf[y_leaf == 3] = 'Binge'
    
    y_leaf = np.squeeze(np.array(y_leaf),1)
        
    y_cat = pd.Categorical(y_leaf, categories=['No use', 'Moderate', 'Misuse', 'Binge'], ordered=True)
    y_final = pd.Series(y_cat)
    
    leaf_header = np.array(header)[leaf["var_idx"]].tolist()
    X_leaf = pd.DataFrame(leaf["exog"], columns = leaf_header)
    
    cats_leaf = []
    for i in range(np.size(X_leaf,1)):
        if leaf['var_types'][i] == "cat" or leaf['var_types'][i] == "ord":
            cats_leaf.append(leaf_header[i])    
    
    Xest = pd.get_dummies(X_leaf, columns = cats_leaf, drop_first=True)
    
    fit = model_ologit.fit(Xest, y_final, cov_type='HC0', method = 'bfgs', skip_hessian = False)
    pred = fit.predict(Xest)
    
    leaf_model[leaf['index']] = [fit]
    
    aux_obs = np.empty((1,5))
    aux_obs[0,0] = leaf['index']
    aux_obs[0,1:] = np.reshape(np.array(y_final.value_counts(normalize=True) * 100), (1,4))
    
    obs_probas = np.vstack([obs_probas, aux_obs])
    
    aux_pred = np.empty((1,5))
    aux_pred[0,0] = leaf['index']
    aux_pred[0,1:] = np.mean(pred, axis = 0)*100

    pred_probas = np.vstack([pred_probas, aux_pred])

export = []
for i in sorted(leaf_model):
    export.extend(leaf_model[i])

### Export observed and predicted probabilities tables to Excel
obs_df = pd.DataFrame(obs_probas)
obs_df.to_excel('G:\Mi unidad\Master\Thesis\Document\Output\Tables\Obs_probas.xlsx', sheet_name='raw', index=False)

probas_df = pd.DataFrame(pred_probas)
probas_df.to_excel('G:\Mi unidad\Master\Thesis\Document\Output\Tables\Pred_probas.xlsx', sheet_name='raw', index=False)

### Export estimation results to Latex (two tables)
export1 = export[:len(export)//2]
export2 = export[len(export)//2:]

## Part 1
latex_table = Stargazer(export1)

latex_table.title("Leaves Model Results")
# latex_table.custom_columns('Leaf Index')
latex_table.add_line('Additional controls', ['Yes']*len(export1))
latex_table.dep_var_name = 'Leaf Index'
latex_table.covariate_order(['alcfage', 'mjnfq', 'cocfq', 'smokeqnt_1.0', 
                             'smokeqnt_2.0', 'smokeqnt_2.0', 'smokeqnt_3.0', 'smokeqnt_4.0', 'smokeqnt_5.0', 'smokeqnt_7.0',
                             'sex', 'age_1.0', 'age_2.0', 'age_3.0', 'age_4.0', 'age_5.0',
                             'married_2.0', 'married_3.0', 'married_4.0',
                             'race_2.0', 'race_3.0', 'race_4.0', 'race_5.0', 'race_6.0',
                             'educcat_2.0', 'educcat_3.0', 'educcat_4.0',
                             'emplmnt_1.0', 'emplmnt_2.0', 'emplmnt_3.0', 'emplmnt_4.0', 'emplmnt_5.0', 'emplmnt_6.0',
                             'fincome_2.0', 'fincome_3.0', 'fincome_4.0', 'fincome_5.0', 'fincome_6.0', 'fincome_7.0',
                             'socialsec',
                             'hhkids_1.0', 'hhkids_2.0', 'hhkids_3.0',
                             'mhealth', 'health_2.0', 'health_3.0', 'health_4.0', 'health_5.0',
                             'relserv_2.0', 'relserv_3.0', 'relserv_4.0', 'relserv_5.0', 'relserv_6.0'])

# Change variables names
latex_table.rename_covariates({'age_1.0': '26 to 29 years old',
                              'age_2.0': '30 to 34 years old',
                              'age_3.0': '35 to 49 years old',
                              'age_4.0': '50 to 64 years old',
                              'age_5.0': '65 years old or older',
                              'alcfage': 'Age first alcohol use',
                              'cocfq': 'Cocaine use frequency',
                              'educcat_2.0': 'High school grad',
                              'educcat_3.0': 'Some college/Assoc Deg',
                              'educcat_4.0': 'College graduate',
                              'emplmnt_1.0': 'Unemployed, looking for work',
                              'emplmnt_2.0': 'Disabled',
                              'emplmnt_3.0': 'Keeping house full-time',
                              'emplmnt_4.0': 'In school/training',
                              'emplmnt_5.0': 'Retired',
                              'emplmnt_6.0': 'Unemployed',
                              'fincome_2.0': '\$10,000 - \$19,999',
                              'fincome_3.0': '\$20,000 - \$29,999',
                              'fincome_4.0': '\$30,000 - \$39,999',
                              'fincome_5.0': '\$40,000 - \$49,999',
                              'fincome_6.0': '\$50,000 - \$74,999',
                              'fincome_7.0': '\$75,000 or more',
                              'health_2.0': 'Fair',
                              'health_3.0': 'Good',
                              'health_4.0': 'Very good',
                              'health_5.0': 'Excellent',
                              'hhkids_1.0': 'One',
                              'hhkids_2.0': 'Two',
                              'hhkids_3.0': 'Three or more',
                              'married_2.0': 'Widowed',
                              'married_3.0': 'Divorced or Separated',
                              'married_4.0': 'Never been married',
                              'mhealth': 'Mental health score',
                              'mjnfq': 'Marijuana use frequency',
                              'race_2.0': 'Non-Hisp Afr Am',
                              'race_3.0': 'Non-Hisp Native',
                              'race_4.0': 'Non-Hisp Asian',
                              'race_5.0': 'Non-Hisp more than one race',
                              'race_6.0': 'Hispanic',
                              'relserv_2.0': '1 to 2',
                              'relserv_3.0': '3 to 5',
                              'relserv_4.0': '6 to 24',
                              'relserv_5.0': '25 to 52',
                              'relserv_6.0': 'More than 52',
                              'sex': 'Sex (Female = 1)',
                              'smokeqnt_1.0': 'Less than 1',
                              'smokeqnt_2.0': 'One',
                              'smokeqnt_3.0': '2 to 5',
                              'smokeqnt_4.0': '6 to 15',
                              'smokeqnt_5.0': '16 to 25',
                              'smokeqnt_6.0': '26 to 35',
                              'smokeqnt_7.0': 'More than 35',
                              'socialsec': 'Receives SS/RR payments (Yes = 1)'})

with open('G:\Mi unidad\Master\Thesis\Document\Output\Tables\Res_leaves1.tex','w') as file:
    file.write(latex_table.render_latex())



## Part 2
latex_table = Stargazer(export2)

latex_table.title("Leaves Model Results (continued)")
# latex_table.custom_columns('Leaf Index')
latex_table.add_line('Additional controls', ['Yes']*len(export2))
latex_table.dep_var_name = 'Leaf Index'
latex_table.covariate_order(['alcfage', 'mjnfq', 'cocfq', 'smokeqnt_1.0', 
                             'smokeqnt_2.0', 'smokeqnt_2.0', 'smokeqnt_3.0', 'smokeqnt_4.0', 'smokeqnt_5.0', 'smokeqnt_7.0',
                             'sex', 'age_1.0', 'age_2.0', 'age_3.0', 'age_4.0', 'age_5.0',
                             'married_2.0', 'married_3.0', 'married_4.0',
                             'race_2.0', 'race_3.0', 'race_4.0', 'race_5.0', 'race_6.0',
                             'educcat_2.0', 'educcat_3.0', 'educcat_4.0',
                             'emplmnt_1.0', 'emplmnt_2.0', 'emplmnt_3.0', 'emplmnt_4.0', 'emplmnt_5.0', 'emplmnt_6.0',
                             'fincome_2.0', 'fincome_3.0', 'fincome_4.0', 'fincome_5.0', 'fincome_6.0', 'fincome_7.0',
                             'socialsec',
                             'hhkids_1.0', 'hhkids_2.0', 'hhkids_3.0',
                             'mhealth', 'health_2.0', 'health_3.0', 'health_4.0', 'health_5.0',
                             'relserv_2.0', 'relserv_3.0', 'relserv_4.0', 'relserv_5.0', 'relserv_6.0'])

# Change variables names
latex_table.rename_covariates({'age_1.0': '26 to 29 years old',
                              'age_2.0': '30 to 34 years old',
                              'age_3.0': '35 to 49 years old',
                              'age_4.0': '50 to 64 years old',
                              'age_5.0': '65 years old or older',
                              'alcfage': 'Age first alcohol use',
                              'cocfq': 'Cocaine use frequency',
                              'educcat_2.0': 'High school grad',
                              'educcat_3.0': 'Some college/Assoc Deg',
                              'educcat_4.0': 'College graduate',
                              'emplmnt_1.0': 'Unemployed, looking for work',
                              'emplmnt_2.0': 'Disabled',
                              'emplmnt_3.0': 'Keeping house full-time',
                              'emplmnt_4.0': 'In school/training',
                              'emplmnt_5.0': 'Retired',
                              'emplmnt_6.0': 'Unemployed',
                              'fincome_2.0': '\$10,000 - \$19,999',
                              'fincome_3.0': '\$20,000 - \$29,999',
                              'fincome_4.0': '\$30,000 - \$39,999',
                              'fincome_5.0': '\$40,000 - \$49,999',
                              'fincome_6.0': '\$50,000 - \$74,999',
                              'fincome_7.0': '\$75,000 or more',
                              'health_2.0': 'Fair',
                              'health_3.0': 'Good',
                              'health_4.0': 'Very good',
                              'health_5.0': 'Excellent',
                              'hhkids_1.0': 'One',
                              'hhkids_2.0': 'Two',
                              'hhkids_3.0': 'Three or more',
                              'married_2.0': 'Widowed',
                              'married_3.0': 'Divorced or Separated',
                              'married_4.0': 'Never been married',
                              'mhealth': 'Mental health score',
                              'mjnfq': 'Marijuana use frequency',
                              'race_2.0': 'Non-Hisp Afr Am',
                              'race_3.0': 'Non-Hisp Native',
                              'race_4.0': 'Non-Hisp Asian',
                              'race_5.0': 'Non-Hisp more than one race',
                              'race_6.0': 'Hispanic',
                              'relserv_2.0': '1 to 2',
                              'relserv_3.0': '3 to 5',
                              'relserv_4.0': '6 to 24',
                              'relserv_5.0': '25 to 52',
                              'relserv_6.0': 'More than 52',
                              'sex': 'Sex (Female = 1)',
                              'smokeqnt_1.0': 'Less than 1',
                              'smokeqnt_2.0': 'One',
                              'smokeqnt_3.0': '2 to 5',
                              'smokeqnt_4.0': '6 to 15',
                              'smokeqnt_5.0': '16 to 25',
                              'smokeqnt_6.0': '26 to 35',
                              'smokeqnt_7.0': 'More than 35',
                              'socialsec': 'Receives SS/RR payments (Yes = 1)'})

with open('G:\Mi unidad\Master\Thesis\Document\Output\Tables\Res_leaves2.tex','w') as file:
    file.write(latex_table.render_latex())



