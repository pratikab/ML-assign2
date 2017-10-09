import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math

def main():

	# Get training file name from the command line
	traindatafile = sys.argv[1];
	# For how many iterations do we wish to execute SCD?
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
	# For dual SCD, you will need to maintain d_alpha and w
	# Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
	w =np.zeros(d);
	d_alpha = np.zeros((n,));

	# We will take a timestamp after every "spacing" iterations
	time_elapsed = np.zeros(math.ceil(n_iter/spacing));
	tick_vals = np.zeros(math.ceil(n_iter/spacing));
	obj_val = np.zeros(math.ceil(n_iter/spacing));

	tick = 0;

	ttot = 0.0;
	t_start = datetime.now();
	numTrain = n;
	for t in range(n_iter):		
		### Doing dual SCD ###

		# Choose a random coordinate from 1 to n
		i_rand = random.randint(0,n-1);
		qii = np.inner(Xtr[i_rand].T,Xtr[i_rand]);
		g = Ytr[i_rand] * (np.matmul(Xtr[i_rand],w)) - 1;
		# Store the old and compute the new value of alpha along that coordinate
		d_alpha_old = d_alpha[i_rand];
		d_alpha[i_rand] = min(max(d_alpha_old-(g/qii),0),1);

		# Update the model - takes only O(d) time!
		w = w + (d_alpha[i_rand] - d_alpha_old)*Ytr[i_rand]*Xtr[i_rand];

		# Take a snapshot after every few iterations
		# Take snapshots after every spacing = 5000 or so SCD iterations since they are fast
		if t%spacing == 0:
			# Stop the timer - we want to take a snapshot
			t_now = datetime.now();
			delta = t_now - t_start;
			time_elapsed[tick] = ttot + delta.total_seconds();
			ttot = time_elapsed[tick];
			tick_vals[tick] = tick;
			loss = np.expand_dims(Ytr, axis=1)*np.matmul(Xtr,np.expand_dims(w, axis=1));
			C = (1-loss) * np.sign(np.maximum(0, 1-loss));
			hinge_loss = np.sum(C);
			u = 0.5*np.linalg.norm(w)
			obj_val[tick] =  u + hinge_loss;
			print("iter = ", t, " val = ",obj_val[tick]);
			# obj_val[tick] = ...; # Calculate the objective value f(w) for the current model w^t
			tick = tick+1;
			# Start the timer again - training time!
			t_start = datetime.now();
			
		# w_final = w.toarray();
		# np.save("model_SCD.npy", w_final);
	count = 0;
	for i in range (0,numTrain):
		y_h = np.matmul(Xtr[i],w.T);
		if y_h*Ytr[i] > 0:
			count += 1;
	print (count/numTrain);
if __name__ == '__main__':
    main()