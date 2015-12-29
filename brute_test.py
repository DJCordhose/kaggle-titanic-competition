"""
Brute force testing of many combinations of features
and several classifiers
"""

import pandas as pd
import numpy as np
import seaborn as sns
#import pprint
###pd.options.display.width = 180
import matplotlib.pyplot as plt
#import pylab as plt
import sklearn as skl
from sklearn import tree
import itertools


# ------------------------------------------------------------------------------------------
def main():
    # Read data
    df = pd.read_table('../titanic/train.csv', sep=",")
    # print(df.head())

    # ------------------------------------------------------------------------------------------

    # Store 'in sample' and  'out of sample' errors: arrays for result df
    E_in = []
    E_out = []
    Model_name = []
    Model_id = []
    iFeatures = []
    Features = []

    # ------------------------------------------------------------------------------------------
    # Preprocessing data
    
    # Set all features
    df['Age'].fillna(0, inplace=True)
    df['Pclass'].fillna(0, inplace=True)
    df['Fare'].fillna(0., inplace=True)
    df['SibSp'].fillna(0., inplace=True)
    df['Parch'].fillna(0., inplace=True)

    df['Sex'].fillna('no', inplace=True)
    df['sex_'] = df['Sex'].map( {'female': 0, 'male': 1, 'no': 2} ).astype(int)

    df['Embarked'].fillna('N', inplace=True)
    df['embarked_'] = df['Embarked'].map( {'N': 0., 'C': 1., 'S': 2., 'Q': 3.} ).astype(float)


    # -------------------------------------------------
    # Slightly more advanced feature extraction

    df['Cabin'].fillna('no', inplace=True)
    def prep_cabin(row):
        res = 0
        if row.lower().find('a')>=0:
            return 1.
        elif row.lower().find('b')>=0:
            return 2.
        elif row.lower().find('c')>=0:
            return 3.
        elif row.lower().find('d')>=0:
            return 4.
        elif row.lower().find('e')>=0:
            return 5.
        elif row.lower().find('f')>=0:
            return 6.
        elif row.lower().find('g')>=0:
            return 7.
        elif row.lower().find('h')>=0:
            return 8.
        return res
    
    df['cabin_'] = df['Cabin'].apply(lambda r: prep_cabin(r))
    #print(df[['Cabin','cabin_']].head(20))
    #exit()

    df['Name'].fillna('no', inplace=True)
    def prep_name(row):
        res = 0
        if row.lower().find('miss.')>=0:
            return 1.
        elif row.lower().find('mrs.')>=0:
            return 2.
        elif row.lower().find('mr.')>=0:
            return 3.
        elif row.lower().find('master')>=0:
            return 4.
        return res
    
    df['name_'] = df['Name'].apply(lambda r: prep_name(r))
    #print(df[['Name','name_']].head(20))
    #exit()

    # -----


    


    # ---
    # Metrics for knn:
    from sklearn import preprocessing

    #scaler = preprocessing.StandardScaler().fit_transform(df['Fare'].values)
    scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(df['Fare'].astype(float).values)
    df['fare_'] = pd.Series(scale)
    #print(df[['Fare','fare_']].head(20))
    
    # ---
    scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(df['Age'].astype(float).values)
    df['age_'] = pd.Series(scale)
    #print(df[['Age','age_']].head(20))
    #exit()

    # ---
    #scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(df['Pclass'].astype(int).values)
    scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(df['Pclass'].astype(float).values)
    df['pclass_'] = pd.Series(scale)
    #print(df[['Pclass','pclass_']].head(20))
    
    # ---
    scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(df['Parch'].astype(float).values)
    df['parch_'] = pd.Series(scale)
    #print(df[['Parch','parch_']].head(20))

    # ---
    scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(df['SibSp'].astype(float).values)
    df['sibsp_'] = pd.Series(scale)
    #print(df[['SibSp','sibsp_']].head(20))

    # ---
    scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(df['embarked_'].astype(float).values)
    df['embarked_'] = pd.Series(scale)
    #print(df[['Embarked','embarked_']].head(20))

    # ---
    scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(df['cabin_'].astype(float).values)
    df['cabin_'] = pd.Series(scale)
    print(df[['Cabin','cabin_']].head(20))

    # ---
    scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(df['name_'].astype(float).values)
    df['name_'] = pd.Series(scale)
    #print(df[['Name','name_']].head(20))

    #exit()

    # ---

    #feature_names = np.array(['sex_', 'Fare'])  #['Pclass', 'Fare', ])
    #feature_names = np.array(['sex_', 'Fare', 'Age', 'Parch', 'SibSp'])  #['Pclass', 'Fare', ])
    #feature_names = np.array(['sex_', 'fare_', 'Age', 'Parch', 'SibSp'])  #['Pclass', 'Fare', ])
    #feature_names = np.array(['sex_', 'fare_', 'age_', 'Pclass', 'Parch', 'SibSp'])  #['Pclass', 'Fare', ])
    #feature_names = np.array(['sex_', 'fare_', 'age_', 'pclass_', 'Parch', 'SibSp'])  #['Pclass', 'Fare', ])
    #feature_names = np.array(['sex_', 'fare_', 'age_', 'pclass_', 'parch_', 'SibSp'])  #['Pclass', 'Fare', ])
    #feature_names = np.array(['sex_', 'fare_', 'age_', 'pclass_', 'parch_', 'SibSp','embarked_'])  #['Pclass', 'Fare', ])
    #feature_names = np.array(['sex_', 'Fare', 'Age', 'Pclass', 'Parch', 'SibSp','embarked_'])  #['Pclass', 'Fare', ])
    #feature_names = np.array(['sex_', 'fare_', 'pclass_', 'parch_', 'sibsp_']) 
    #feature_names = np.array(['sex_', 'fare_', 'pclass_']) 
    #feature_names = np.array(['sex_', 'fare_']) 
    #feature_names = np.array(['sex_', 'fare_', 'age_', 'pclass_', 'parch_', 'sibsp_','embarked_'])  
    ###feature_names = np.array(['sex_', 'fare_', 'age_', 'pclass_', 'parch_', 'sibsp_'])  
    #feature_names = np.array(['sex_', 'fare_', 'pclass_','name_','cabin_']) 
    feature_names = np.array(['sex_', 'fare_', 'age_', 'pclass_', 'name_', 'cabin_', 'parch_', 'sibsp_'])  
    
    label_name = ['Survived']

    # Generate feature combinations
    print(feature_names)
    fc = []
    flen = len(feature_names)
    for fl in np.arange(flen)+1:
        c = itertools.combinations(feature_names, fl)
        # check:
        for s in c:
            #print(s)
            fc.append(np.array(s))

            
    Feature_list = fc
    lFl = len(fc)
    #for f in Feature_list:
    #    print(f) 

    #exit()

    feat_c = 0
    # Run over all feature combinations on different classifiers
    for fn in Feature_list:
        # KNN model: Run over some NN 
        for nn in (1,5,10,15):
            e_in,e_out = knn(df,label_name,fn,lFl,feat_c,n_neighbors=nn)
            Model_name.append('KNN k'+str(nn))
            E_in.append(e_in)
            E_out.append(e_out)
            iFeatures.append(feat_c)
            Features.append(fn)

        # DTree
        e_in,e_out = dtree(df,label_name,fn,lFl,feat_c)
        Model_name.append('DTree')
        E_in.append(e_in)
        E_out.append(e_out)
        iFeatures.append(feat_c)
        Features.append(fn)

        # SVM
        #e_in,e_out = svm(df,label_name,fn,lFl,feat_c,C=10.)
        e_in,e_out = svm(df,label_name,fn,lFl,feat_c,C=1000.)
        #e_in,e_out = svm(df,label_name,fn,lFl,feat_c,C=100000.)
        Model_name.append('SVM')
        E_in.append(e_in)
        E_out.append(e_out)
        iFeatures.append(feat_c)
        Features.append(fn)

        # RF
        e_in,e_out = rf(df,label_name,fn,lFl,feat_c,n_estimators=100)
        #e_in,e_out = rf(df,label_name,fn,lFl,feat_c,n_estimators=1000)
        Model_name.append('RF')
        E_in.append(e_in)
        E_out.append(e_out)
        iFeatures.append(feat_c)
        Features.append(fn)

        # adaboost
        e_in,e_out = adaboost(df,label_name,fn,lFl,feat_c,n_estimators=100)
        Model_name.append('adaboost')
        E_in.append(e_in)
        E_out.append(e_out)
        iFeatures.append(feat_c)
        Features.append(fn)

        feat_c += 1
        pass

    # ------------------------------------------------------------------------------------------
    # Fill results dataframe
    
    Model_id = np.arange(len(E_in))
    #print(Feature_list)
    #print(Model_id)
    #print(Model_name)
    #print(E_in)
    #print(E_out)

    #modeldf = pd.DataFrame({'Name': Model_name, 'E_in': E_in, 'E_out': E_out, 'iFeatures': iFeatures},index=Model_id)
    modeldf = pd.DataFrame({'Name': Model_name, 'E_in': E_in, 'E_out': E_out, 'Features': Features},index=Model_id)

    # Sort by best performing models
    modeldf.sort(columns=['E_out','E_in'],ascending=[1,1],inplace=True)
    #print(modeldf.head())
    
    modeldf.to_csv('result_brute_test.csv')
    
    # ------------------------------------------------------------------------------------------
    # Print out best performing models
    #nbest = 10
    ##best_model = modeldf.ix[modeldf['E_out'].argmin()]
    #best_model = modeldf.ix[modeldf['E_out'].argsort().values[:nbest]]

    print(modeldf.head(20)) # print best models
    #print(best_model)
    #for i in best_model['iFeatures'].values:
    #    print(i,Feature_list[i])

    # ------------------------------------------------------------------------------------------
    # Plot performance vs models
    plt.rc('text', usetex=True)
    line_ein = plt.plot(Model_id,np.array(E_in)*100.,label=r'$E_{in}$ #("in sample")')
    line_eout = plt.plot(Model_id,np.array(E_out)*100.,label=r'$E_{out}$ ("out of sample")')
    plt.title('')
    plt.xlabel('Model Id')
    plt.ylabel('Error Rate (\%)')
    plt.legend(handles=[line_ein, line_eout],labels=['',''])
    #plt.show()
    plt.savefig('chart_brute_test.png')
    
# ------------------------------------------------------------------------------------------
# KNN
def knn(df,label_name,feature_names,features_len,ifeat,n_neighbors=5):
    from sklearn import neighbors
    print('---------------------------------------------------')
    print(ifeat,features_len,'KNN, nn=',n_neighbors,' features:',feature_names)
    df_train_Y = df[label_name]
    train_Y = df_train_Y.values.ravel()  # turn from 2D to 1D

    df_train_X = df[feature_names]
    train_X = df_train_X.values

    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf = clf.fit(train_X,train_Y)
    # output = clf.predict(train_X)
    E_in = round(1.-clf.score(train_X, train_Y),5) # 'in sample' error
    #print('\tE_in :',E_in)

    # -----
    # Kfold as estimator for 'out of sample' error
    kf=skl.cross_validation.KFold(n=len(train_X), n_folds=5)
    cv_scores=skl.cross_validation.cross_val_score(clf, train_X, y=train_Y, cv=kf)
    E_out = round(1.-np.mean(cv_scores),5)
    #print("\tE_out:",E_out)

    return E_in,E_out

# ------------------------------------------------------------------------------------------
# Decision Tree
def dtree(df,label_name,feature_names,features_len,ifeat):
    from sklearn import tree
    print('---------------------------------------------------')
    print(ifeat,features_len,'DTree, features:',feature_names)
    df_train_Y = df[label_name]
    train_Y = df_train_Y.values.ravel()  # turn from 2D to 1D

    df_train_X = df[feature_names]
    train_X = df_train_X.values

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_X,train_Y)
    # output = clf.predict(train_X)
    E_in = round(1.-clf.score(train_X, train_Y),5) # 'in sample' error
    #print('\tE_in :',E_in)

    # -----
    # Kfold as estimator for 'out of sample' error
    kf=skl.cross_validation.KFold(n=len(train_X), n_folds=5)
    cv_scores=skl.cross_validation.cross_val_score(clf, train_X, y=train_Y, cv=kf)
    E_out = round(1.-np.mean(cv_scores),5)
    #print("\tE_out:",E_out)

    return E_in,E_out

# ------------------------------------------------------------------------------------------
# Support Vector Machine
def svm(df,label_name,feature_names,features_len,ifeat,C=1.0):
    from sklearn import svm
    print('---------------------------------------------------')
    print(ifeat,features_len,'SVM, features:',feature_names)
    df_train_Y = df[label_name]
    train_Y = df_train_Y.values.ravel()  # turn from 2D to 1D

    df_train_X = df[feature_names]
    train_X = df_train_X.values

    clf = svm.SVC(C=C)
    clf = clf.fit(train_X,train_Y)
    # output = clf.predict(train_X)
    E_in = round(1.-clf.score(train_X, train_Y),5) # 'in sample' error
    #print('\tE_in :',E_in)

    # -----
    # Kfold as estimator for 'out of sample' error
    kf=skl.cross_validation.KFold(n=len(train_X), n_folds=5)
    cv_scores=skl.cross_validation.cross_val_score(clf, train_X, y=train_Y, cv=kf)
    E_out = round(1.-np.mean(cv_scores),5)
    #print("\tE_out:",E_out)

    return E_in,E_out

# ------------------------------------------------------------------------------------------
# Random Forest
def rf(df,label_name,feature_names,features_len,ifeat,n_estimators=100):
    from sklearn.ensemble import RandomForestClassifier
    print('---------------------------------------------------')
    print(ifeat,features_len,'Random Forest, features:',feature_names)
    df_train_Y = df[label_name]
    train_Y = df_train_Y.values.ravel()  # turn from 2D to 1D

    df_train_X = df[feature_names]
    train_X = df_train_X.values

    clf =RandomForestClassifier(n_estimators=n_estimators)
    clf = clf.fit(train_X,train_Y)
    # output = clf.predict(train_X)
    E_in = round(1.-clf.score(train_X, train_Y),5) # 'in sample' error
    #print('\tE_in :',E_in)

    # -----
    # Kfold as estimator for 'out of sample' error
    kf=skl.cross_validation.KFold(n=len(train_X), n_folds=5)
    cv_scores=skl.cross_validation.cross_val_score(clf, train_X, y=train_Y, cv=kf)
    E_out = round(1.-np.mean(cv_scores),5)
    #print("\tE_out:",E_out)

    return E_in,E_out


# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
# ------------------------------------------------------------------------------------------
# adaboost
def adaboost(df,label_name,feature_names,features_len,ifeat,n_estimators=100):
    # TODO: just copied from RF, needs real code
    from sklearn.ensemble import RandomForestClassifier
    print('---------------------------------------------------')
    print(ifeat,features_len,'Adaboost, features:',feature_names)
    df_train_Y = df[label_name]
    train_Y = df_train_Y.values.ravel()  # turn from 2D to 1D

    df_train_X = df[feature_names]
    train_X = df_train_X.values

    clf =RandomForestClassifier(n_estimators=n_estimators)
    clf = clf.fit(train_X,train_Y)
    # output = clf.predict(train_X)
    E_in = round(1.-clf.score(train_X, train_Y),5) # 'in sample' error
    #print('\tE_in :',E_in)

    # -----
    # Kfold as estimator for 'out of sample' error
    kf=skl.cross_validation.KFold(n=len(train_X), n_folds=5)
    cv_scores=skl.cross_validation.cross_val_score(clf, train_X, y=train_Y, cv=kf)
    E_out = round(1.-np.mean(cv_scores),5)
    #print("\tE_out:",E_out)

    return E_in,E_out

# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
