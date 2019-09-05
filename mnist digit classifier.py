import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data=pd.read_csv(r'C:\Users\walru\OneDrive\Documents\python\Kaggle Digit Recognizer\train.csv')
test=pd.read_csv(r'C:\Users\walru\OneDrive\Documents\python\Kaggle Digit Recognizer\test.csv')

labels=data['label']
y=pd.get_dummies(labels)
y2=y[5]
x=data.drop(['label'], axis=1)
features=[]

for col in x.columns:
    features.append(tf.feature_column.numeric_column(col))

x_train, x_test, y_train, y_test=train_test_split(x, labels, test_size=0.1)

input_func=tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=64, num_epochs=None, shuffle=True)

model=tf.estimator.DNNClassifier(feature_columns=features, hidden_units=[256,64,32], n_classes=10, optimizer=tf.train.AdamOptimizer(1e-4))

model.train(input_fn=input_func, steps=20000)

input_func_test=tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=len(x_test), num_epochs=1, shuffle=False)

predictions=list(model.predict(input_fn=input_func_test))

final_preds=[]

for pred in predictions:
	final_preds.append(pred['class_ids'][0])

print(classification_report(y_test, final_preds))

ImageId=np.arange(1,28001)
input_func_predict=tf.estimator.inputs.pandas_input_fn(x=test, batch_size=len(test), num_epochs=1, shuffle=False)
Label_output=list(model.predict(input_fn=input_func_predict))
Label=[]
for pred in Label_output:
	Label.append(pred['class_ids'][0])
submission={'ImageId':ImageId, 'Label':Label}
sub_df=pd.DataFrame(submission)

	


