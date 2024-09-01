from src.data_cleaner import DataCleaner

# Load sample data
import pandas as pd
data = pd.read_csv('data/sample_data.csv')

# Initialize the Data Cleaner
cleaner = DataCleaner(data)

# Clean the data
clean_data = cleaner.clean()

# Save the cleaned data
clean_data.to_csv('data/cleaned_data.csv', index=False)
