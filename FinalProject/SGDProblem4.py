#David Shagam Task4
import random
import pandas
import numpy as np
from sklearn import metrics
import scipy as sp
import sklearn as sk
from sklearn import linear_model
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

def getSGDHyperParams():
	hyperParameters= {
	'normalization': random.choice(["l1","l2","max","none"]),
	'loss':random.choice(["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]),
	'penalty':random.choice(['none','l2','l1','elasticnet']),
	'alpha':random.uniform(0.00001,0.1),
	'l1_ratio':random.uniform(0,1),
	'fit_intercept':random.choice([True,False]),
	'n_iter':random.randrange(1,100,1),
	'shuffle':random.choice([True,False]),
	'epsilon':random.uniform(0.001,1),
	}
	return hyperParameters


#run random search
bestVal = -1
bestModel = None
for i in range(100):
	hp = getSGDHyperParams()
	workTrain=trainMatrix
	workDev= devMatrix
	if hp['normalization'] != "none":
		workTrain = sk.preprocessing.normalize(trainMatrix, norm=hp['normalization'], axis=0)
		workDev = sk.preprocessing.normalize(devMatrix, norm=hp['normalization'], axis=0)

	mySGD = linear_model.SGDRegressor(loss=hp['loss'], penalty=hp['penalty'], alpha=hp['alpha'], l1_ratio=hp['l1_ratio'], fit_intercept=hp['fit_intercept'], n_iter=hp['n_iter'], shuffle=hp['shuffle'], epsilon=hp['epsilon'])
	mySGD.fit(workTrain, trainY)
	predictions = mySGD.predict(workDev)
	score = metrics.mean_squared_error(devY, predictions)
	hp['dev_accuracy'] = score
	log = pandas.DataFrame(hp, index=[0])
	print("MFE: %f"% score)

	if i == 0:
		log.to_csv("SGD_data.csv", mode='a')
	else:
		log.to_csv("SGD_data.csv", mode='a', header=False)

	if score<bestVal or bestVal == -1:
		bestVal = score
		bestModel = mySGD

if bestModel != None:
	pickle.dump(bestModel, open("BestSGD.p", "wb"))


#gradient descent method EG Program 1 style
