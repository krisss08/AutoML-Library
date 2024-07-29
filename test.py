from config import configurations
from automl import AutoML
import pandas as pd

# train_data = pd.read_csv('C:/Users/chand/Downloads/train_titanic.csv')
# test_data = pd.read_csv('C:/Users/chand/Downloads/test_titanic.csv')

train_data = pd.read_csv("/Users/krishivijayanand/Downloads/MAD/AutoML/train_titanic.csv")
test_data = pd.read_csv("/Users/krishivijayanand/Downloads/MAD/AutoML/test_titanic.csv")


x_train, y_train = train_data.drop(columns=['Survived'], axis = 1), train_data['Survived']

x_test, y_test = test_data.drop(columns=['Survived'], axis = 1), test_data['Survived']

print(x_train.shape)

# # train and predict
ml = AutoML(configurations)
if not configurations.get("run_id"): 
    ml.fit(x_train, y_train)

print(ml.predict(x_test))