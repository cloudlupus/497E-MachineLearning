#David Shagam task 4
import random
import pandas
import numpy as np
from sklearn import metrics
import scipy as sp
import sklearn as sk
from sklearn.svm import SVR
import pickle

#Loading Data

sparseTrainXF = open("./task4/train.sparseX")
trainYF = open("./task4/train.RT")
sparseDevXF = open("./task4/dev.sparseX")
devYF = open("./task4/dev.RT")
config = open("./task4/task4.config")

trainSparse = np.loadtxt(sparseTrainXF)
trainY = np.loadtxt(trainYF)
devSparse = np.loadtxt(sparseDevXF)
devY = np.loadtxt(devYF)

#print(trainSparse)
#print(trainSparse.max(axis=0))

#Load config
configDict={}
for lines in config:
	key, val = lines.split()
	configDict[key] = int(val);

sparseTrainXF.close()
trainYF.close()
sparseDevXF.close()
devYF.close()
config.close()
#print(configDict)

#Make data no longer a sparse matrix aka make it a dense
trow=[]
tcol=[]
tval=[]

for entry in trainSparse:
	trow.append(entry[0])
	tcol.append(entry[1])
	tval.append(entry[2])

drow=[]
dcol=[]
dval=[]
for entry in devSparse:
	drow.append(entry[0])
	dcol.append(entry[1])
	dval.append(entry[2])
trainSparse = None
devSparse = None

trainMatrix = sp.sparse.coo_matrix((tval,(trow,tcol)), shape=(configDict["N_TRAIN"], configDict["D"])).tocsc()
devMatrix = sp.sparse.coo_matrix((dval,(drow,dcol)), shape=(configDict["N_DEV"], configDict["D"])).tocsc()





#Neural Network method

def getSVRHyperParams():
	hyperParameters= {
	'normalization': random.choice(["l1","l2","max","none"]),
	'C':random.uniform(0.0, 5.0),
	'epsilon':random.uniform(0.001,0.5),
	'kernel':random.choice(["linear","poly","rbf","sigmoid"]),
	'degree':random.randrange(2,10,1),
	'gamma': random.choice(["auto", random.uniform(0.0001,0.1)]),
	'coef0': random.uniform(0.0,1.0),
	'shrinking': random.choice([True,False]),
	'max_iter': random.randrange(1000,10000,100)
	}
	return hyperParameters


#run random search
bestVal = -1
bestModel = None
for i in range(100):
	hp = getSVRHyperParams()
	workTrain=trainMatrix
	workDev= devMatrix
	if hp['normalization'] != "none":
		workTrain = sk.preprocessing.normalize(trainMatrix, norm=hp['normalization'], axis=0)
		workDev = sk.preprocessing.normalize(devMatrix, norm=hp['normalization'], axis=0)

	mySVR = SVR(C=hp['C'], epsilon=hp['epsilon'], kernel=hp['kernel'], degree=hp['degree'], gamma=hp['gamma'], coef0=hp['coef0'], shrinking=hp['shrinking'], max_iter=hp['max_iter'])
	mySVR.fit(workTrain, trainY)
	predictions = mySVR.predict(workDev)
	score = metrics.mean_squared_error(devY, predictions)
	hp['dev_accuracy'] = score
	log = pandas.DataFrame(hp, index=[0])
	print("MFE: %f"% score)

	if i == 0:
		log.to_csv("SVR_data.csv", mode='a')
	else:
		log.to_csv("SVR_data.csv", mode='a', header=False)

	if score<bestVal or bestVal == -1:
		bestVal = score
		bestModel = mySVR
		pickle.dump(bestModel, open("BestSVR.p", "wb"))


#gradient descent method EG Program 1 style

