
---

**`src/data_cleaner.py`**:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def handle_missing_values(self):
        """Handle missing values by filling with mean, median, or mode."""
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                if self.data[column].dtype == np.number:
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                else:
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)

    def handle_outliers(self):
        """Handle outliers by capping values at the 1st and 99th percentiles."""
        for column in self.data.select_dtypes(include=[np.number]).columns:
            q1 = self.data[column].quantile(0.01)
            q99 = self.data[column].quantile(0.99)
            self.data[column] = np.clip(self.data[column], q1, q99)

    def normalize_data(self):
        """Normalize numerical features to standard scale."""
        scaler = StandardScaler()
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[num_cols] = scaler.fit_transform(self.data[num_cols])

    def encode_categorical(self):
        """Encode categorical features with label encoding."""
        for column in self.data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])

    def clean(self):
        """Run all cleaning functions."""
        self.handle_missing_values()
        self.handle_outliers()
        self.normalize_data()
        self.encode_categorical()
        return self.data
