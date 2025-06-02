import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class CSVPreprocessor:
    def __init__(self, input_path, output_path = "cleaned_output.csv", column_string = "sys;dia;hr;spo2;features"):
        """
        input_path: path to the original CSV file
        output_path: path to save the processed file
        column_string: string with feature names separated by semicolons (';')
                       the last feature is considered the target (categorical)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.column_names = column_string.split(';')
        self.input_columns = self.column_names[: -1]  # input features
        self.output_column = self.column_names[-1]   # target variable
        self.encoder = LabelEncoder()
        self.df = None

    def load_and_split(self):
        # Load and split by ';'
        raw_df = pd.read_csv(self.input_path)
        self.df = raw_df.iloc[:, 0].str.split(';', expand = True)
        self.df.columns = self.column_names

    def convert_inputs(self):
        # Convert input features to float32
        for col in self.input_columns:
            self.df[col] = self.df[col].astype(np.float32)

    def encode_output(self):
        # Encode the target variable
        self.df[f"{self.output_column}_encoded"] = self.encoder.fit_transform(self.df[self.output_column])
        self.df.drop(columns = [self.output_column], inplace = True)

    def save(self):
        # Save the final file
        self.df.to_csv(self.output_path, index = False)
        print(f"The file was saved successfully: {self.output_path}")

    def process(self):
        # Run the full preprocessing pipeline
        self.load_and_split()
        self.convert_inputs()
        self.encode_output()
        self.save()
