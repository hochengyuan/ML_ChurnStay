import pandas as pd
import numpy as np
import csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import linear_model
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

def main():
    names = ['COLLEGE', 'INCOME', 'OVERAGE', 'LEFTOVER', 'HOUSE', 'HANDSET_PRICE',
            'OVER_15MINS_CALLS_PER_MONTH', 'AVERAGE_CALL_DURATION', 'REPORTED_SATISFACTION',
            'REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN','LEAVE']

    train_df = pd.read_csv('train.csv', delimiter=',')
    test_df = pd.read_csv('test.csv', delimiter=',')
    combine = [train_df, test_df]
    # train_df = train_df.drop(['COLLEGE', 'AVERAGE_CALL_DURATION', 'OVER_15MINS_CALLS_PER_MONTH', ], axis=1)
    # train_df = train_df.drop([ 'COLLEGE', 'REPORTED_USAGE_LEVEL', 'CONSIDERING_CHANGE_OF_PLAN', 'REPORTED_SATISFACTION'], axis=1)

    for dataset in combine:
        dataset['COLLEGE'] = dataset['COLLEGE'].map( {'one': 1, 'zero': 0} ).astype(int)
        dataset['REPORTED_SATISFACTION'] = dataset['REPORTED_SATISFACTION'].map( {'very_unsat': 1, 'unsat': 2, 'avg' : 3, 'sat' : 4, 'very_sat' : 5} ).astype(int)
        dataset['REPORTED_USAGE_LEVEL'] = dataset['REPORTED_USAGE_LEVEL'].map( {'very_little': 1, 'little': 2, 'avg' : 3, 'high' :4, 'very_high' :5} ).astype(int)
        dataset['CONSIDERING_CHANGE_OF_PLAN'] = dataset['CONSIDERING_CHANGE_OF_PLAN'].map( {'no': 1, 'never_thought': 2, 'perhaps':3, 'considering': 4, 'actively_looking_into_it': 5} ).astype(int)

    # train_df['SURVEY'] = train_df['REPORTED_SATISFACTION'] * train_df['REPORTED_USAGE_LEVEL'] * train_df['CONSIDERING_CHANGE_OF_PLAN']
    #
    #
    # # pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='summer_r')
    #
    # # print(train_df[['SURVEY', 'LEAVE']].groupby(['SURVEY'], as_index= False).mean().sort_values(by='LEAVE', ascending=False))
    # train_df['INCOME'] = pd.qcut(train_df['INCOME'], 4, labels = [1, 2, 3, 4])
    # print(train_df[['INCOME', 'LEAVE']].groupby(['INCOME'], as_index=False).mean().sort_values(by='INCOME', ascending=True))
    #
    # train_df['HOUSE'] = pd.qcut(train_df['HOUSE'], 4, labels = [1, 2, 3, 4])
    # print(train_df[['HOUSE', 'LEAVE']].groupby(['HOUSE'], as_index=False).mean().sort_values(by='HOUSE', ascending=True))
    #
    # train_df['HANDSET_PRICE'] = pd.cut(train_df['HANDSET_PRICE'], 2, labels = [1, 2])
    # print(train_df[['HANDSET_PRICE', 'LEAVE']].groupby(['HANDSET_PRICE'], as_index=False).mean().sort_values(by='HANDSET_PRICE', ascending=True))
    # #
    # # train_df['AVERAGE_CALL_DURATION'] = pd.qcut(train_df['AVERAGE_CALL_DURATION'], 3, labels = [1, 2, 3])
    # # print(train_df[['AVERAGE_CALL_DURATION', 'LEAVE']].groupby(['AVERAGE_CALL_DURATION'], as_index=False).mean().sort_values(by='AVERAGE_CALL_DURATION', ascending=True))
    # #
    # # train_df['OVER_15MINS_CALLS_PER_MONTH'] = pd.qcut(train_df['OVER_15MINS_CALLS_PER_MONTH'], 3, labels = [1, 2, 3])
    # # print(train_df[['OVER_15MINS_CALLS_PER_MONTH', 'LEAVE']].groupby(['OVER_15MINS_CALLS_PER_MONTH'], as_index=False).mean().sort_values(by='OVER_15MINS_CALLS_PER_MONTH', ascending=True))
    # #
    # #
    # # # train_df['OVERAGE'] = pd.qcut(train_df['OVERAGE'], 5)
    # # # print(train_df[['OVERAGE', 'LEAVE']].groupby(['OVERAGE'], as_index=False).mean().sort_values(by='OVERAGE', ascending=True))
    # #
    # train_df['OVERAGE'] = pd.cut(train_df['OVERAGE'], 4, labels = [1, 2, 3, 4])
    # print(train_df[['OVERAGE', 'LEAVE']].groupby(['OVERAGE'], as_index=False).mean().sort_values(by='OVERAGE', ascending=True))
    #
    #
    # train_df['LEFTOVER'] = pd.cut(train_df['LEFTOVER'], 4, labels = [1, 2 ,3,4])
    # print(train_df[['LEFTOVER', 'LEAVE']].groupby(['LEFTOVER'], as_index=False).mean().sort_values(by='LEFTOVER', ascending=True))
    #
    #
    # train_df['WEALTH'] = train_df['HOUSE'] +  2 * train_df['INCOME']
    #
    # train_df['WEALTH'] = pd.qcut(train_df['WEALTH'], 4 , labels = [1, 2, 3, 4])
    #


    train_df = train_df.drop([ 'COLLEGE', "REPORTED_SATISFACTION", "REPORTED_USAGE_LEVEL" ,"CONSIDERING_CHANGE_OF_PLAN" ,'OVER_15MINS_CALLS_PER_MONTH', 'AVERAGE_CALL_DURATION'], axis=1)
    # print(train_df[['WEALTH', 'LEAVE']].groupby(['WEALTH'], as_index=False).mean().sort_values(by='WEALTH', ascending=True))

    # # print(train_df.head())
    X = train_df.drop("LEAVE", axis=1)
    y = train_df["LEAVE"]
    y = y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1800, random_state=10)
    # # X_train, X_test = feature_normalization(X_train, X_test)
    # scaler = preprocessing.MinMaxScaler()
    # scaler.fit_transform(X_train)
    # scaler.fit_transform(X_test)
    #


    random_forest = RandomForestClassifier(n_estimators=150)
    random_forest .fit(X_train, y_train)
    acc_random_forest = random_forest.score(X_test, y_test) * 100

    svc = SVC()
    svc.fit(X_train, y_train)
    acc_svc = svc.score(X_test, y_test) * 100

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    acc_knn = knn.score(X_test, y_test) * 100

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    acc_log = logreg.score(X_test, y_test) * 100



    print(random_forest.feature_importances_)



    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    acc_sgd = round(sgd.score(X_test, y_test) * 100, 2)

    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    acc_gbc = round(gbc.score(X_test, y_test) * 100, 2)

    models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'SGD' ,'GBC'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_sgd, acc_gbc]})

    models.sort_values(by='Score', ascending=False)


    print(models)
    #
    #
    # # C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # # gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # # kernel=['rbf','linear']
    # # hyper={'kernel':kernel,'C':C,'gamma':gamma}
    # # gd=GridSearchCV(estimator=SVC(),param_grid=hyper,verbose=True)
    # # gd.fit(X_train,y_train)
    # # print(gd.best_score_)
    # # print(gd.best_estimator_)
    # # print(train_df.groupby(['COLLEGE','LEAVE'])['LEAVE'].count())


    # predictions = random_forest.predict(test_df)
    #
    # fh = open("test_custom.csv" , "w")
    # lines_of_text = ["ID,LEAVE\n"]
    # index = 0
    # for i in np.nditer(predictions.T, order='C'):
    #     lines_of_text.append(str(index) + "," + str(i) + "\n")
    #     index += 1;
    # fh.writelines(lines_of_text)
    # fh.close()


if __name__ == '__main__':
    main()
