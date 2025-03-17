import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from src.config import TARGET

def cramers_v(confusion_matrix):
    """computing Cramér's V (for categorical features)"""
    
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def compute_categorical_associations(data):
    """Computing correlation matrix"""
    
    df = data.copy()
    df[TARGET] = df[TARGET].astype('str')
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    associations = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
    
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 == col2:
                associations.loc[col1, col2] = 1.0  # Perfect correlation with itself
            else:
                contingency_table = pd.crosstab(df[col1], df[col2])
                associations.loc[col1, col2] = cramers_v(contingency_table)
    
    return associations



