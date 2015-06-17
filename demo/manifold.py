# Visualize

import math
import pylab as pl
from sklearn import datasets,manifold
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

import numpy as np


def NonlinearMap(TrnData1, TrnData2, TstData):

	if 1:
		Model = manifold.LocallyLinearEmbedding(n_neighbors=50, n_components=17, eigen_solver='auto', method='modified')
		Model.fit(TrnData1)
		joblib.dump(Model, "LLE_Modified.pkl")
		print "Model 1 fitted."
	else:
		Model = joblib.load("LLE_Modified.pkl")

	Ytrn = Model.transform(TrnData2)
	Ytst = Model.transform(TstData)
	print "data transformed by Model1."

	#################################################################
	if 0:
		if 0:
			Model = manifold.LocallyLinearEmbedding(n_neighbors=50, n_components=17, eigen_solver='auto', method='standard')
			Model.fit(TrnData1)
			joblib.dump(Model, "LLE_Standard.pkl")
			print "Model 2 fitted."
		else :
			Model = joblib.load("LLE_Standard.pkl")

		Ytrn = Model.transform(TrnData2)
		Ytst = Model.transform(TstData)
		print "data transformed by Model2."

		if 0:
			if 0:
				Model = manifold.TSNE(n_components=17, perplexity=50.0)
				Model.fit(TrnData1)
				joblib.dump(Model, "TSNE.pkl")
				print "Model 3 fitted."
			else :
				Model = joblib.load("TSNE.pkl")

			Ytrn = Model.fit_transform(TrnData2)
			Ytst = Model.fit_transform(TstData)
			print "data transformed by Model3."

		if 0:
			if 0:
				Model = manifold.MDS(n_components=17, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=-1, random_state=None, dissimilarity='euclidean')
				Model.fit(TrnData1)
				joblib.dump(Model, "MDS.pkl")
				print "Model 4 fitted."
			else :
				Model = joblib.load("MDS.pkl")

			Ytrn = Model.fit_transform(TrnData2)
			Ytst = Model.fit_transform(TstData)
			print "data transformed by Model4."

		if 0:
			if 0:
				Model = manifold.SpectralEmbedding(n_components=17,n_neighbors=50)
				Model.fit(TrnData1)
				joblib.dump(Model, "SE.pkl")
				print "Model 5 fitted."
			else :
				Model = joblib.load("SE.pkl")

			Ytrn = Model.fit_transform(TrnData2)
			Ytst = Model.fit_transform(TstData)
			print "data transformed by Model5."


	return Ytrn, Ytst


if __name__ == "__main__":

	#initialise data set
	Path = "./"
	Data = np.genfromtxt(Path+"Train_Otto.csv", delimiter=',')
	Lbl = np.genfromtxt(Path+"Label_Otto.csv", delimiter=',')
	print Data.shape
	print Lbl.shape


	#Data = stats.zscore(Data,axis=0,ddof=0)
	Mean = np.mean(Data,axis=0)
	Std = np.std(Data,axis=0)
	Data = Data - np.kron(np.ones((Data.shape[0],1)), Mean)
	Data = np.divide(Data , np.kron(np.ones((Data.shape[0],1)), Std))
	print "Train data collected."


	Rind = np.random.permutation(Data.shape[0])
	TrnNo1 = 300 #5000
	TrnNo2 = 500 #50000
	TstNo  = 500 #10000
	TrnData1 = Data[Rind[1:TrnNo1]][:]
	TrnData2 = Data[Rind[1:TrnNo2]][:]
	TrnLbl = Lbl[Rind[1:TrnNo2]]

	TstData = Data[Rind[TrnNo2+1:TrnNo2+TstNo]][:]
	TstLbl = Lbl[Rind[TrnNo2+1:TrnNo2+TstNo]]

	#np.save("Data_Otto_5000.npz", TrnData1)
	#np.save("Label_Otto_5000.npz", Lbl[Rind[1:TrnNo1]])

	print TstData.shape

	Ytrn, Ytst = NonlinearMap(TrnData1, TrnData2, TstData)

	#Ytrn = np.concatenate((TrnData2,Ytrn),axis=1)
	#Ytst = np.concatenate((TstData,Ytst),axis=1)

	print Ytrn.shape
	print Ytst.shape

	#pl.scatter(Y[:, 0], Y[:, 1],c = Lbl,cmap=pl.cm.hot, marker="x")
	#pl.show()

	from validation.score import prob_to_cls, calc_accuracy
	model = RandomForestClassifier(n_estimators = 150, criterion='gini', max_features=10, max_depth=45, min_samples_split=2, min_samples_leaf=1, n_jobs=1, verbose=0)
	model.fit(Ytrn, TrnLbl)
	PrdCls = model.predict(Ytst)
	#PrdCls = prob_to_cls(predictions)
	accuracy = PrdCls==TstLbl
	accuracy = np.mean(accuracy)
	print "Accuracy: "+str(accuracy)
	hist, bin_edges = np.histogram(TstLbl,9, (1,9), normed =False)
	print hist
	print "finished"