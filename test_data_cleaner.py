import unittest
import pandas as pd
from src.data_cleaner import DataCleaner

class TestDataCleaner(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing."""
        self.sample_data = pd.DataFrame({
            'age': [25, 35, None, 40, 60, 200],
            'income': [50000, 60000, None, 80000, 100000, 1500000],
            'gender': ['male', 'female', None, 'female', 'male', 'female']
        })
        self.cleaner = DataCleaner(self.sample_data)

    def test_handle_missing_values(self):
        """Test if missing values are correctly handled."""
        self.cleaner.handle_missing_values()
        self.assertFalse(self.cleaner.data.isnull().values.any())

    def test_handle_outliers(self):
        """Test if outliers are handled correctly."""
        self.cleaner.handle_outliers()
        self.assertTrue(self.cleaner.data['age'].max() < 100)

    def test_normalize_data(self):
        """Test if data normalization works."""
        self.cleaner.normalize_data()
        self.assertAlmostEqual(self.cleaner.data['income'].mean(), 0, delta=1)

    def test_encode_categorical(self):
        """Test if categorical features are encoded correctly."""
        self.cleaner.encode_categorical()
        self.assertTrue(self.cleaner.data['gender'].dtype == 'int64')

if __name__ == '__main__':
    unittest.main()
