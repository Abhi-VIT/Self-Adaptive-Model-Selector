import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def debug():
    print("Starting Debug")
    try:
        df = pd.DataFrame({
            'num': [1.0, 2.0, np.nan, 4.0],
            'cat': ['A', 'B', np.nan, 'A'],
            'target': [1, 0, 1, 0]
        })
        
        X = df.drop(columns=['target']).copy()
        
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns
        
        print(f"Num cols: {num_cols}")
        print(f"Cat cols: {cat_cols}")
        
        if len(num_cols) > 0:
            print("Imputing numeric...")
            num_imputer = SimpleImputer(strategy='median')
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
            
        if len(cat_cols) > 0:
            print("Imputing categorical...")
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
            
        print("Debug Success")
        print(X)
    except Exception as e:
        print(f"DEBUG FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
