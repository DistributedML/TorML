from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np
import sys
import utils
import pdb

if __name__ == "__main__":

	victim_csv = sys.argv[1]

	# Inverted attacked dataset
	data = utils.load_dataset("credit1")
	X, y = data['X'], data['y']
	
	testdata = utils.load_dataset("credittest")
	Xvalid, yvalid = testdata['X'], testdata['y']

	# Train the optimal classifier
	clf = SGDClassifier(loss="hinge", penalty="l2")
	clf.fit(X, y)

	real_hat = clf.predict(Xvalid)

	victim_data = np.loadtxt(victim_csv, delimiter=',')
	victim_model = victim_data[0,:]

	victim_model = np.array(victim_model)
	victim_hat = np.sign(np.dot(Xvalid, victim_model))

	print("Agreement is %.3f" % (sum(victim_hat == real_hat) / float(real_hat.shape[0])))


