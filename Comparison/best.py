from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings("ignore")
iris=load_breast_cancer()
X = iris.data
Y = iris.target
scaled_x=preprocessing.scale(X)
seed = 7
models = []
lr = LogisticRegression(dual=False,max_iter=1000,tol=0.0001)
models.append(('Logistic Regression', lr))
knn = KNeighborsClassifier(n_neighbors=9,algorithm='ball_tree',weights='distance')
models.append(('K-Nearest Neighbours', knn))
dt = DecisionTreeClassifier()
dt1 = tree.DecisionTreeClassifier()
models.append(('Decision Tree', dt))
nb = GaussianNB()
models.append(('Naive Bayes', nb))
svm = SVC(C=10,kernel='linear',tol=0.09)
models.append(('SVM', svm))
ann = MLPClassifier(hidden_layer_sizes=(50,45),activation='logistic',learning_rate_init=0.001)
models.append(('Neural Net', ann))
p = Perceptron()
models.append(('Perceptron', p))
bc = BaggingClassifier(n_estimators=50,bootstrap=True,bootstrap_features=False)
models.append(('Bagging', bc))
abc = AdaBoostClassifier(n_estimators=200,algorithm='SAMME')
models.append(('AdaBoost', abc))
gbc = GradientBoostingClassifier(n_estimators=200,max_depth=3)
models.append(('Gradient Boosting', gbc))
rf = RandomForestClassifier(n_estimators=30,bootstrap=True,min_impurity_split=0.00001)
models.append(('Random Forest', rf))
dnn = MLPClassifier(hidden_layer_sizes=(50,45,35,30,28,25,20),activation='identity',learning_rate_init=0.001)
models.append(('Deep Learning', dnn))

results = []
results1 = []
names = []
final = []
print(' ')
print('Result of Different Classifiers on Breast Cancer DataSet:')
print(' ')
scoring = 'accuracy'
scoring1 = 'precision_weighted'
print('-----------------------------------------------')
print('| {:^20} | {:} | {:} |'.format("Model", "Accuracy", "Precision"))
print('-----------------------------------------------')
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X, Y, cv=kfold, scoring=scoring)
    cv_results1 = model_selection.cross_val_score(model,X, Y, cv=kfold, scoring=scoring1)
    results.append(cv_results)
    results1.append(cv_results1)
    names.append(name)
    print('| {:^20} | {:0.4f}% | {:0.5f}% |'.format(name, cv_results.mean()*100, cv_results1.mean()*100))
    final.append(cv_results.mean()*100)
print('-----------------------------------------------')

results = [z* 100 for z in results]
fig = plt.figure()
fig.suptitle('Accuracy of Different Classifiers on Breast Cancer DataSet (Best Parameters)')
ax = fig.add_subplot(111)
plt.boxplot(results, showmeans=True)
ax.set_xticklabels(names, rotation='vertical')
plt.show()

results1 = [z* 100 for z in results]
fig = plt.figure()
fig.suptitle('Precision of Different Classifiers on Breast Cancer DataSet (Best Parameters)')
ax = fig.add_subplot(111)
plt.boxplot(results, showmeans=True)
ax.set_xticklabels(names, rotation='vertical')
plt.show()
