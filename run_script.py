import pandas as pd
import joblib
import preproccessing as p
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

print("test model for by load")

dataTest = pd.read_csv("airline-test-samples.csv")
dataTest1 = pd.read_csv("airline-test-samples1.csv")
dataTest2 = pd.read_csv("airline-test-samples1.csv")
filenamelinear_model ='filenamelinear_model.sav'
filenameLinearRegression='filenameLinearRegression.sav'


filenameDicision = "finalModelDicision.sav"
filenameKNearest = "finalModelKNearest.sav"
filenameRandomForest = "finalModelRandomForest.sav"

xTest ,yTest = p.preprocessingfunTest(dataTest)
xTest1 ,yTest1 = p.preprocessingfunTestRegCorr(dataTest1)
xTest11 ,yTest11 = p.preprocessingfunTestRegK(dataTest2)

loadDicisionModel = joblib.load(filenameDicision)
k = joblib.load(filenameKNearest)
f = joblib.load(filenameRandomForest)

lp=joblib.load(filenamelinear_model)

l=joblib.load(filenameLinearRegression)
poly_features = PolynomialFeatures(degree=4)

p1 = lp.predict(poly_features.fit_transform(xTest1))
p2= l.predict(xTest11)

pr=loadDicisionModel.predict(xTest)
pr1=k.predict(xTest)
pr2=f.predict(xTest)
print(loadDicisionModel.score(xTest,yTest))
print(k.score(xTest,yTest))
print(f.score(xTest,yTest))

print('------------------------------------')


print(r2_score(yTest1,p1))
print(r2_score(yTest11,p2))
# print(p1.score(xTest1,yTest1))
# print(p2.score(xTest11 ,yTest11))