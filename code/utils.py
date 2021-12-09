import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import cross_validate, cross_val_predict, KFold, StratifiedKFold

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import f1_score, auc, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt
import seaborn as sns


# Define numeric and categorical features 
# extracted from the MIMIC-IV database

EMR_NUM_FEATURES = ['age', 'hours_in_hosp_before_intubation', 'weight', 'height',
                    'co2_total_max', 'co2_total_avg', 'co2_total_min', 'ph_max', 'ph_avg', 'ph_min', 
                    'lactate_max', 'lactate_avg', 'lactate_min', 'pao2fio2ratio', 
                    'heart_rate_max', 'heart_rate_avg', 'heart_rate_min', 'mbp_max', 'mbp_avg', 'mbp_min',
                    'mbp_ni_max', 'mbp_ni_avg', 'mbp_ni_min', 'resp_rate_max', 'resp_rate_avg', 'resp_rate_min',
                    'temp_max','temp_avg', 'temp_min', 'spo2_max', 'spo2_avg', 'spo2_min',
                    'glucose_max', 'glucose_avg', 'glucose_min', 
                    'vasopressin', 'epinephrine', 'dobutamine', 'norepinephrine', 'phenylephrine', 'dopamine', 'count_of_vaso',
                    'fio2_max', 'fio2_avg', 'fio2_min', 'peep_max', 'peep_avg', 'peep_min',
                    'plateau_pressure_max', 'plateau_pressure_avg', 'plateau_pressure_min',
                    'rrt', 'sinus_rhythm', 'neuroblocker', 'congestive_heart_failure', 'cerebrovascular_disease', 'dementia',
                    'chronic_pulmonary_disease', 'rheumatic_disease', 'mild_liver_disease', 
                    'diabetes_without_cc', 'diabetes_with_cc', 'paraplegia', 
                    'renal_disease', 'malignant_cancer', 'severe_liver_disease',
                    'metastatic_solid_tumor', 'aids', 
                    'SOFA', 'respiration', 'coagulation', 'liver', 'cardiovascular', 'cns', 'renal',
                    'apsiii', 'hr_score', 'mbp_score', 'temp_score', 'resp_rate_score', 
                    'pao2_aado2_score', 'hematocrit_score','wbc_score', 'creatinine_score', 
                    'uo_score', 'bun_score', 'sodium_score', 'albumin_score', 
                    'bilirubin_score', 'glucose_score', 'acidbase_score', 'gcs_score', 
                   ]

EMR_CAT_FEATURES = ['admission_location', 'insurance', 'language', 
                    'ethnicity', 'marital_status', 'gender']



def get_X_and_y(df, features=None, label="over72h", incl_cluster_id=False):
    if features:
        if features=="numeric":
            X = df[EMR_NUM_FEATURES]
        elif features=="categorical":
            X = df[EMR_CAT_FEATURES]
        else:
            X = df[features]
    else:
        X = df[EMR_NUM_FEATURES + EMR_CAT_FEATURES]
        
    if incl_cluster_id:
        X = pd.concat([X, df.cluster], axis=1)
        
    y = df[label].values
    return X, y


def define_preprocessor(features, scale=True, one_hot=True):
    if scale:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ])
    if one_hot:    
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
    else:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ])
    
    numeric_features = [f for f in features if f in EMR_NUM_FEATURES]
    categorical_features = [f for f in features if f in EMR_CAT_FEATURES]
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor
        

def cluster_by_severity(df, pca=None, severity_scores=None, trained_on="mimic"):
    if severity_scores==None:
        severity_scores = [ 'SOFA', 'respiration', 'coagulation', 'liver', 
                           'cardiovascular', 'renal', 'cns','apsiii', 'hr_score', 
                           'mbp_score', 'temp_score', 'resp_rate_score', 'pao2_aado2_score', 
                           'hematocrit_score','wbc_score', 'creatinine_score', 'uo_score', 
                           'bun_score', 'sodium_score', 'albumin_score', 'bilirubin_score', 
                           'glucose_score', 'acidbase_score', 'gcs_score'
                          ]
        
    print("Using %d severity scores..." % len(severity_scores))
    
    # Impute missig values
    preprocessor = define_preprocessor(severity_scores, scale=False)
    X = preprocessor.fit_transform(df[severity_scores])
    
    # Fit PCA
    if pca==None:
        print("Fitting PCA...");
        pca = PCA(n_components=3)
        pca.fit(X)
     
    # Apply transformations
    X_pca = pca.transform(X)
    
    if trained_on=="mimic":
        # Assign clusters
        labels = np.full((X_pca.shape[0]), 4)
        labels[X_pca[:, 1] > 0.5 * X_pca[:, 0] - 24.75] = 3
        labels[X_pca[:, 1] > 0.5 * X_pca[:, 0] - 0.5] = 2
        labels[X_pca[:, 1] > 0.5 * X_pca[:, 0] + 16] = 1
    elif trained_on=="eicu":
        # Assign clusters
        labels = np.full((X_pca.shape[0]), 4)
        labels[-X_pca[:, 1] > 0.5 * X_pca[:, 0] - 24.75] = 3
        labels[-X_pca[:, 1] > 0.5 * X_pca[:, 0] - 0.5] = 2
        labels[-X_pca[:, 1] > 0.5 * X_pca[:, 0] + 16] = 1
    
    df = pd.concat([df, pd.DataFrame(X_pca, columns=["pc1", "pc2", "pc3"])], axis=1)
    df["cluster"] = labels
    print(df.cluster.value_counts())
    
    return df, pca
    

def benchmark_cv(model, X, y, head="clf"):
    print('_' * 80)
    print()
    print("Model training: ")
    
    t0 = time()
    
    if head == "clf":
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        return y_proba
    elif head == "reg":
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        
        return y_pred
        
        
def benchmark_cv_score(model, X, y, head="clf", verbose=True):
    print('_' * 80)
    print()
    print("Model training: ")
    
    if head == "clf":
        try:
            print(model['classifier'])
        except:
            print(model)
    
        scoring = {
            "Precision" : "precision_macro",
            "Recall" : "recall_macro",
            "F1" : "f1_macro", 
            "ROC AUC" : "roc_auc", 
            "PR AUC" : "average_precision"
        }  
        t0 = time()
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_validate(model, X, y, n_jobs=-1, cv=cv, scoring=scoring)
        train_time = time() - t0
        
    elif head == "reg":
        try:
            print(model['regressor'])
        except:
            print(model)
            
        scores = {
            'RMSE': [] , 
            'Pearson': [], 
            'Spearman': [], 
#             'Kendall': []
        }
        t0 = time()
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        for train_ind, val_ind in cv.split(X,y):
            model.fit(X.iloc[train_ind], y[train_ind])
            y_pred = model.predict(X.iloc[val_ind])
            scores['RMSE'].append(mean_squared_error(y[val_ind], y_pred, squared=False))
            scores['Pearson'].append(pearsonr(y[val_ind], y_pred)[0])
            scores['Spearman'].append(spearmanr(y[val_ind], y_pred)[0])
#             scores['Kendall'].append(kendalltau(y[val_ind], y_pred)[0]) 
        train_time = time() - t0
            
    if verbose:
        print("train time: %0.3fs" % train_time)
        print()
        for score in scores:
            if "time" not in score:
                print("Average %s: %0.2f (+/- %0.2f)" % (score, np.mean(scores[score]), np.std(scores[score]) * 2)) 
     
    return scores
    
    
def show_values_on_bars(ax, orient="v", space=0.4):
    if orient == "v":
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = int(p.get_height())
            ax.text(_x, _y, value, ha="center") 
    elif orient == "h":
        for p in ax.patches:
            _x = p.get_x() + p.get_width() + float(space)
            _y = p.get_y() + p.get_height() - p.get_height() / 4
            value = round(p.get_width(), 2)
            ax.text(_x, _y, value, ha="left", fontsize=14)    
            
            
def evaluate_model(y, y_proba, class_names, string, thresh=None, show_plots=True, digits=2, save_figures=False, filename=""):
    n_outputs = y_proba.shape[1]
    if n_outputs > 2:
        average = 'macro'
    else:
        average = 'binary'
    
    # Generate predictions
    if thresh:
        y_pred = np.where(y_proba[:,-1] > thresh, 1, 0)
    else:
        if n_outputs == 1:
            y_pred = np.where(y_proba > 0.5, 1, 0)
        else:
            y_pred = np.argmax(y_proba, axis=1)
            
    print("Model evaluation on the %s set" % string)
    print()
    
    # Classification report
    print("Classification report:")
    print(classification_report(y, y_pred, digits=digits)) 
    
    # Plot confusion matrix
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.figure();
    sns.heatmap(confusion_matrix(y, y_pred, normalize="true"), 
                annot=confusion_matrix(y, y_pred),
                annot_kws={'fontsize' : 16}, fmt="d",
                cmap="Blues", cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix");
    
    if save_figures:
        plt.savefig(filename + "_CM.png", bbox_inches='tight', dpi=300, transparent=True, pad_inches=0);
    
    if show_plots:
        if n_outputs > 1:
            y_dummy = pd.get_dummies(y, drop_first=False).values
            
        # Plot ROC curves
        plt.figure();
        sns.lineplot(x=[0, 1], y=[0, 1], color=sns.color_palette()[0], lw=2, linestyle='--', label="Chance")
        if len(class_names) == 2:
            fpr, tpr, _ = roc_curve(y, y_proba[:,-1])
            roc_auc = roc_auc_score(y, y_proba[:,-1])
            sns.lineplot(x=fpr, y=tpr, lw=3, color=sns.color_palette()[1], 
                         label="AUC = %0.2f" % roc_auc)
        else:
            for i in range(n_outputs):
                fpr, tpr, _ = roc_curve(y_dummy[:,i], y_proba[:,i])
                roc_auc = roc_auc_score(y_dummy[:,i], y_proba[:,i], multi_class="ovr")
                sns.lineplot(x=fpr, y=tpr, lw=3, color=sns.color_palette()[1 + i], 
                             label=class_names[i] + " (AUC = %0.2f)" % roc_auc)

        plt.xlim([-0.01, 1.0])
        plt.ylim([-0.01, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right", fontsize=16);
        
        if save_figures:
            plt.savefig(filename + "_ROC.png", bbox_inches='tight', dpi=300, pad_inches=0);

        # Plot precision-recall curves
        plt.figure();
        
        if n_outputs == 1:
            prec, rec, _ = precision_recall_curve(y, y_proba)
            pr_auc = auc(rec, prec)
            sns.lineplot(x=rec, y=prec, lw=3, color=sns.color_palette()[1], 
                         label="AUC = %0.2f" % pr_auc)
        else:
            for i in range(n_outputs):
                prec, rec, _ = precision_recall_curve(y_dummy[:,i], y_proba[:,i])
                pr_auc = auc(rec, prec)
                sns.lineplot(x=rec, y=prec, lw=3, color=sns.color_palette()[1 + i], 
                             label=class_names[i] + " (AUC = %0.2f)" % pr_auc)

        plt.xlim([-0.01, 1.0])
        plt.ylim([-0.01, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend(loc="lower right", fontsize=16);
        
        if save_figures:
            plt.savefig(filename + "_PR.png", bbox_inches='tight', dpi=300, pad_inches=0); 
          
    return y_pred
            
            
    