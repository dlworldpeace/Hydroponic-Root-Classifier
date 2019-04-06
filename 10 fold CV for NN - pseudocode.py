from sklearn.model_selection import KFold
import statistics

X = ##dataset of the root images
y = ## labels of the roots
kf = RepeatedKFold(n_splits = 10, n_repeats = 1, random_state = 42)
cv_err_hu = [[],[],[],[],[],[],[],[],[],[]]
hidden_units = [1,2,3,4,5,6,7,8,9,10]

for train_index, test_index in kf.split(X):
        print("train:", train_index, "validation:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        i = 0
        for num_hu in hidden_units:
            ##include the code here to build a model using the X_train and the y_train
            ## for num_hu hidden units
            ## find pred = predictions using the X_test
        
            sq_residuals = (y_test - pred)^2 ##obtain squared residuals
            cv_error_hu[i].append(statistics.mean(sq_residuals)) ##MSE
            i+=1

cv_err = []
for ele in cv_err_hu:
    cv_err.append(statistics.mean(ele))

##decide how many hidden units should be included to obtain the lowest cv_err
##hidden_units[i] would be the best number of hidden units
for i in range(0,10):
    if cv_err[i] == min(cv_err):
        print(i)
