import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import matplotlib.pyplot as plt
import math
def main():

	# Get training file name from the command line
	traindatafile = sys.argv[1];
	# For how many iterations do we wish to execute GD?
	n_iter = int(sys.argv[2]);
	# After how many iterations do we want to timestamp?
	spacing = int(sys.argv[3]);	
	# The training file is in libSVM format
	tr_data = load_svmlight_file(traindatafile);
	Xtr = tr_data[0]; # Training features in sparse format
	Ytr = tr_data[1]; # Training labels

	# We have n data points each in d-dimensions
	n, d = Xtr.get_shape();

	# The labels are named 1 and 2 in the data set. Convert them to our standard -1 and 1 labels
	Ytr = 2*(Ytr - 1.5);
	Ytr = Ytr.astype(int);
	# Optional: densify the features matrix.
	# Warning: will slow down computations
	Xtr = Xtr.toarray();

	# Initialize model
	# For primal GD, you only need to maintain w
	# Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
	# w = csr_matrix((1, d));
	w = np.zeros([1, d]);
	# We will take a timestamp after every "spacing" iterations
	time_elapsed = np.zeros(math.ceil(n_iter/spacing));
	tick_vals = np.zeros(math.ceil(n_iter/spacing));
	obj_val = np.zeros(math.ceil(n_iter/spacing));

	tick = 0;

	ttot = 0.0;
	t_start = datetime.now();

	numTrain = 240000;
	# y = np.expand_dims(x, axis=0)
	for t in range(0,n_iter):
		### Doing primal GD ###
		
		# Compute gradient
		g = np.zeros([1, d]);
		# print(g);
		# g = g+w;
		g = np.copy(w);
		y = Ytr[0:numTrain];
		y = np.expand_dims(y, axis=1)
		y_h = np.matmul(Xtr[0:numTrain],w.T);
		loss = np.ravel(np.multiply(y,y_h));
		hinge_loss = 0;
		A = np.expand_dims( -1*Ytr[0:numTrain], axis=1);
		B = np.multiply(A,Xtr[0:numTrain]);
		C = (1-loss) * np.sign(np.maximum(0, 1-loss));
		D = B * np.expand_dims(np.sign(np.maximum(0, 1-loss)),axis = 1);
		E = np.sum(D,axis = 0).ravel();
		hinge_loss = np.sum(C);
		g= g + E;
		# for i in range(0,numTrain):		
		# 	if loss[i] < 1:
		# 		hinge_loss += (1-loss[i]);
		# 		tg = (-1*Ytr[i])*Xtr[i]
		# 		g += tg;
		# print (hinge_loss, np.sum(C));
		# g.reshape(1,d); # Reshaping since model is a row vector
		# print(E);
		# print(g);
		# g = g+w;
		# Calculate step lenght. Step length may depend on n and t
		eta = 0.0001/(math.sqrt(t+1));
		# Update the model
		# print("++++++");
		w = w - (eta) * g;
		# Use the averaged model if that works better (see [\textbf{SSBD}] section 14.3)
		# wbar = ...;
		
		# Take a snapshot after every few iterations
		# Take snapshots after every spacing = 5 or 10 GD iterations since they are slow
		if t%spacing == 0:
			# Stop the timer - we want to take a snapshot
			t_now = datetime.now();
			delta = t_now - t_start;
			time_elapsed[tick] = ttot + delta.total_seconds();
			ttot = time_elapsed[tick];
			tick_vals[tick] = tick;
			u = 0.5*np.linalg.norm(w)
			obj_val[tick] =  u + hinge_loss; # Calculate the objective value f(w) for the current model w^t or the current averaged model \bar{w}^t
			print ("t = ",t, " f = ", obj_val[tick], "  u = ", u , " loss  = " , hinge_loss);
			tick = tick+1;
			# Start the timer again - training time!
			t_start = datetime.now();
		
	# Choose one of the two based on whichever works better for you
	# w_final = w.toarray();
	# w_final = wbar.toarray();
	np.save("model_GD.npy", w);
	count = 0;
	for i in range (0,numTrain):
		y_h = np.matmul(Xtr[i],w.T);
		if y_h*Ytr[i] > 0:
			count += 1;
	print (count/numTrain);
if __name__ == '__main__':
    main()