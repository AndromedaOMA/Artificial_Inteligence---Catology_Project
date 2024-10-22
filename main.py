import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd
import xlrd


# Read the Xlsx file
data = pd.read_excel("data_set_cat.xlsx", "Data")
print(data)
# Handle missing data
data.fillna(-1, inplace=True)

print(data)
