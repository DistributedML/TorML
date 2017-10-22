from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np

if __name__ == "main":

	# Inverted attacked dataset
	data = utils.load_dataset("credit1")
	X, y = data['X'], data['y']
	
	testdata = utils.load_dataset("credittest")
	Xvalid, yvalid = testdata['X'], testdata['y']

	# Train the optimal classifier
	clf = SGDClassifier(loss="hinge", penalty="l2")
	clf.fit(X, y)

	clf.predict(Xvalid)

	



