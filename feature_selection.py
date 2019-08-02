from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as pyplot
feature_vectors1, targets1 = load_svmlight_file("trainingdatafileTFIDF.txt")
feature_vectors2, targets2 = load_svmlight_file("trainingdatafileTFIDF.txt")
feature_vectors3, targets3 = load_svmlight_file("trainingdatafileTFIDF.txt")
import warnings
warnings.filterwarnings('ignore')
kvalue = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000,10000]
print ("Calculating and generating the plot:")
multinomialnbf1scores = []
bernoullinbf1scores = []
svcf1scores = []
kneighbourf1scores = []
for i in kvalue:
 cls1 = MultinomialNB()
 X_new1 = SelectKBest(chi2, k=i).fit_transform(feature_vectors1, targets1)
 f1 = cross_val_score(cls1, X_new1, targets1, cv=5, scoring='f1_macro')
 multinomialnbf1scores.append(f1.mean())
for i in kvalue:
 cls2 = BernoulliNB()
 X_new1 = SelectKBest(chi2, k=i).fit_transform(feature_vectors2, targets2)
 f1 = cross_val_score(cls2, X_new1, targets2, cv=5, scoring='f1_macro')
 bernoullinbf1scores.append(f1.mean())
for i in kvalue:
 cls3 = SVC(gamma='auto')
 X_new1 = SelectKBest(chi2, k=i).fit_transform(feature_vectors3, targets3)
 f1 = cross_val_score(cls3, X_new1, targets3, cv=5, scoring='f1_macro')
 svcf1scores.append(f1.mean())
for i in kvalue:
 cls4 = KNeighborsClassifier()
 X_new1 = SelectKBest(chi2, k=i).fit_transform(feature_vectors3, targets3)
 f1 = cross_val_score(cls4, X_new1, targets3, cv=5, scoring='f1_macro')
 kneighbourf1scores.append(f1.mean())
pyplot.figure(figsize=(9,9))
pyplot.plot(kvalue, multinomialnbf1scores,label = "Multinomial Naive Bayes")
pyplot.plot(kvalue, bernoullinbf1scores, label = "Bernoulli Naive Bayes")
pyplot.plot(kvalue, svcf1scores, label = "SVM")
pyplot.plot(kvalue, kneighbourf1scores, label = "KNN")
pyplot.xlabel("K")
pyplot.ylabel("f1_macro (CHI Square)")
pyplot.legend(loc = 'best')
pyplot.show()
print ("Calculating and generating the second plot:")
kvalue = [100, 400, 600, 800, 1000, 1200]
multinomialnbmif1scores = []
bernoullinbmif1scores = []
svcmif1scores = []
kneighbourmif1scores = []
for i in kvalue:
 cls1 = MultinomialNB()
 X_new1 = SelectKBest(mutual_info_classif, k=i).fit_transform(feature_vectors1,targets1)
 f1 = cross_val_score(cls1, X_new1, targets1, cv=5, scoring='f1_macro')
 multinomialnbmif1scores.append(f1.mean())
for i in kvalue:
 cls2 = BernoulliNB()
 X_new1 = SelectKBest(mutual_info_classif, k=i).fit_transform(feature_vectors2,targets2)
 f1 = cross_val_score(cls2, X_new1, targets2, cv=5, scoring='f1_macro')
 bernoullinbmif1scores.append(f1.mean())
for i in kvalue:
 cls3 = SVC(gamma='auto')
 X_new1 = SelectKBest(mutual_info_classif, k=i).fit_transform(feature_vectors3,targets3)
 f1 = cross_val_score(cls3, X_new1, targets3, cv=5, scoring='f1_macro')
 svcmif1scores.append(f1.mean())
for i in kvalue:
 cls4 = KNeighborsClassifier()
 X_new1 = SelectKBest(mutual_info_classif, k=i).fit_transform(feature_vectors3,targets3)
 f1 = cross_val_score(cls4, X_new1, targets3, cv=5, scoring='f1_macro')
 kneighbourmif1scores.append(f1.mean())
pyplot.figure(figsize=(9,9))
pyplot.plot(kvalue, multinomialnbmif1scores,label = "Multinomial Naive Bayes")
pyplot.plot(kvalue, bernoullinbmif1scores, label = "Bernoulli Naive Bayes")
pyplot.plot(kvalue, svcmif1scores, label = "SVM")
pyplot.plot(kvalue, kneighbourmif1scores, label = "KNN")
pyplot.xlabel("K")
pyplot.ylabel("f1_macro (Mutual Information)")
pyplot.legend(loc = 'best')
pyplot.show()