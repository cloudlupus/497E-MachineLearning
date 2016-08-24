import random
import pandas
import numpy as np
from sklearn import metrics
import scipy as sp
import sklearn as sk
from sklearn.svm import SVR
from sklearn import linear_model
import pickle

sparseDevXF = open("./task4/test.sparseX")
config = open("./task4/task4.config")
outputFile = open("task4.predictions","w")

devSparse = np.loadtxt(sparseDevXF)

configDict={}
for lines in config:
	key, val = lines.split()
	configDict[key] = int(val);

drow=[]
dcol=[]
dval=[]
print("begin COO make")
for entry in devSparse:
	drow.append(entry[0])
	dcol.append(entry[1])
	dval.append(entry[2])

print("Making COO to CSC")
devMatrix = sp.sparse.coo_matrix((dval,(drow,dcol)), shape=(53969, configDict["D"])).tocsc()
print("Finished CSC")
print("begin norm")
devMatrix = sk.preprocessing.normalize(devMatrix, norm="max", axis=0)
print("end norm loaind pickle")
model = pickle.load(open("./BestSGD.p","rb"))
print("finished loading pickle, predicting")
predictions = model.predict(devMatrix)

print(predictions)

for val in predictions:
	outputFile.write(str(val))
	outputFile.write("\n")
