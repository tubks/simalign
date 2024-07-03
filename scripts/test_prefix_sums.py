
import numpy as np
def build_prefix_array(sim_array):
	prefix_array = np.zeros(sim_array.shape)
	prefix_array[0,0] = sim_array[0,0]
	for i in range(1,sim_array.shape[0]):
		prefix_array[i,0] = prefix_array[i-1,0]+sim_array[i,0]
	for j in range(1,sim_array.shape[1]):
		prefix_array[0,j] = prefix_array[0,j-1]+sim_array[0,j]
	for i in range(1, sim_array.shape[0]):
		for j in range(1,sim_array.shape[1]):
			prefix_array[i,j]=prefix_array[i-1,j]+prefix_array[i,j-1]-prefix_array[i-1,j-1]+sim_array[i,j]
	# print(prefix_array.shape, prefix_array)
	return prefix_array


def compute_score_from_prefix_array(row_start,row_end, col_start,col_end):
	if row_start==0 and col_start==0:
		print("PREFIX INDICES: ",row_start,row_end, col_start, col_end)
		print("PREFIX SUMS: ",prefix_array[row_end-1,col_end-1])
		prefix_slice = prefix_array[row_start:row_end-1, col_start:col_end-1]
		print("PREFIX SLICE: ", prefix_slice)
		print("PREFIX SLICE SHAPE: ", prefix_slice.shape)
		return prefix_array[row_end-1,col_end-1]
	if row_start==0 and col_start!=0:
		print("PREFIX INDICES: ",row_start,row_end, col_start, col_end)
		print("PREFIX SUMS: ", prefix_array[row_end-1,col_end-1]-prefix_array[row_end-1,col_start-1])
		prefix_slice = prefix_array[row_start-1:row_end-1, col_start-1:col_end-1]
		print("PREFIX SLICE: ", prefix_slice)
		print("PREFIX SLICE SHAPE: ", prefix_slice.shape)
		return prefix_array[row_end-1,col_end-1]-prefix_array[row_end-1,col_start-1]
	if row_start!=0 and col_start==0:
		print("PREFIX INDICES: ",row_start,row_end, col_start, col_end)
		print("PREFIX SUMS: ",prefix_array[row_end-1,col_end-1]-prefix_array[row_start-1,col_end-1])
		prefix_slice = prefix_array[row_start-1:row_end-1, col_start:col_end-1]
		print("PREFIX SLICE: ", prefix_slice)
		print("PREFIX SLICE SHAPE: ", prefix_slice.shape)
		return prefix_array[row_end-1,col_end-1]-prefix_array[row_start-1,col_end-1]
	else:
		print("PREFIX INDICES: ",row_start,row_end, col_start, col_end)
		print("PREFIX SUMS: ",prefix_array[row_end-1, col_end-1], prefix_array[row_start-1,col_start-1], prefix_array[row_end-1,col_end-1]-prefix_array[row_end-1,col_start-1]-prefix_array[row_start-1,col_end-1]+prefix_array[row_start-1,col_start-1])
		prefix_slice = prefix_array[row_start-1:row_end-1, col_start-1:col_end-1]
		print("PREFIX SLICE: ", prefix_slice)
		print("PREFIX SLICE SHAPE: ", prefix_slice.shape)
		return prefix_array[row_end-1,col_end-1]-prefix_array[row_end-1,col_start-1]-prefix_array[row_start-1,col_end-1]+prefix_array[row_start-1,col_start-1]
def W(X,Y): # TODO: use prefix sums
	# print(f'W: {X},{Y}')
	# print("matrix slice:", X[0],X[-1],Y[0],Y[-1])
	if len(X)==1 and len(Y)==1:
		return sim[X[0],Y[0]]
	elif len(X)==1 and len(Y)!=1:
		return np.sum(sim[X[0],Y[0]:Y[-1]])
	elif len(X)!=1 and len(Y)==1:
		return np.sum(sim[X[0]:X[-1],Y[0]])
	else:
		print("SIM SUM: ",np.sum(sim[X[0]:X[-1],Y[0]:Y[-1]]))
		print("SIM SLICE: ",sim[X[0]:X[-1],Y[0]:Y[-1]])
		print("SIM SLICE SHAPE: ", sim[X[0]:X[-1],Y[0]:Y[-1]].shape)
		return compute_score_from_prefix_array(X[0],X[-1],Y[0],Y[-1])
		# return np.sum(sim[X[0]:X[-1],Y[0]:Y[-1]])

if __name__=="__main__":
	sim=np.ones((5,5))
	prefix_array = build_prefix_array(sim)
	print(W([0,2],[1,4]))