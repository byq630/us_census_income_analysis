import re
from sklearn.preprocessing import OrdinalEncoder
from src.config import TARGET, RAW_COLUMN_INFO

def get_column_names():
    """ Adding column names to the raw data based on given information"""
    
    column_names = [line.split(":")[0].strip() for line in RAW_COLUMN_INFO.split("\n") if ":" in line]
    cleaned_columns =  [re.sub(r"\s+", "_", col.lower()) for col in column_names]
    cleaned_columns.append(TARGET)
    
    return cleaned_columns


class CategoricalEncoder:
    def __init__(self, ordered_categories=None, non_ordered_cols=None, int_cols=None):
        """
        Initialises the encoder with specified ordered categories, non-ordered columns, and integer transformation columns.
        
        :param ordered_categories: dict where keys are column names and values are lists defining the order of categories.
        :param non_ordered_cols: list of column names to be encoded without a specific order.
        :param int_cols: list of column names to be converted to integer type.
        """
        self.ordered_categories = ordered_categories or {}
        self.non_ordered_cols = non_ordered_cols or []
        self.int_cols = int_cols or []
        
        # define encoders for ordered (OrdinalEncoder) & non-ordered columns (can use OrdinalEncoder or TargetEncoder, etc.)
        self.encoders = {
            col: OrdinalEncoder(categories=[self.ordered_categories[col]])
            for col in self.ordered_categories
        }
        self.non_ordered_encoder = OrdinalEncoder()
    
    def fit(self, df):
        """
        Fits the encoders on the df.
        
        :param df: df containing the categorical columns to be encoded.
        """
        for col, encoder in self.encoders.items():
            df[col] = df[col].str.lstrip()  # Ensure consistent formatting
            encoder.fit(df[[col]])
        
        if self.non_ordered_cols:
            self.non_ordered_encoder.fit(df[self.non_ordered_cols])
    
    def transform(self, df):
        """
        Transforms the categorical columns in the df using the fitted encoders.
        
        :param df: df containing the categorical columns to be transformed.
        :return: df with transformed categorical values.
        """
        df_transformed = df.copy()
        
        for col, encoder in self.encoders.items():
            df_transformed[col] = encoder.transform(df_transformed[[col]])
        
        if self.non_ordered_cols:
            df_transformed[self.non_ordered_cols] = self.non_ordered_encoder.transform(df_transformed[self.non_ordered_cols])
        
        for col in self.int_cols:
            df_transformed[col] = df_transformed[col].astype('int')
        
        return df_transformed
    
    def fit_transform(self, df):
        """
        Fits the encoders and transforms the df in one step.
        
        :param df: df containing the categorical columns to be encoded.
        :return: transformed df.
        """
        self.fit(df)
        return self.transform(df)