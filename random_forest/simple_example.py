


from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from utils.loading import get_training_data, get_testing_data
from validation.crossvalidate import SampleCrossValidator
    
train, labels, features = get_training_data()
test, features = get_testing_data()

rfc = RandomForestClassifier(n_estimators = 50)
rfc.fit(train, labels)
rfc.predict_proba(test)



