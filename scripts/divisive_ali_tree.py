from divisive_alignment import Ncut


def align_create_trees(S,T,sim,prefix_array,sentence_tree,sent_id,depth,parent):
	node_tag = f"('{S[0]}-{S[-1]}','{T[0]}-{T[-1]}')"
	node_id = f"{sent_id}.{depth}_{S[0]}-{S[-1]}_{T[0]}-{T[-1]}"
	if not parent:	# root node has no parent
		sentence_tree.create_node(node_tag,node_id)
	else:
		sentence_tree.create_node(node_tag,node_id,parent=parent)
	if len(S)==1 or len(T)==1:
		return S,T # terminate the procedure
	minNcut = 2
	X,Y = S,T
	# loop over the indices that are the potential cutting points
	for i in range(1,len(S)):
		for j in range(1,len(T)):
			A = S[:i]
			B = T[:j]
			A_bar = S[i:]
			B_bar = T[j:]
			newNcut = Ncut(A,B, A_bar, B_bar,sim,prefix_array)
			if newNcut<minNcut:
				minNcut = newNcut
				X,Y = A, B
				X_bar, Y_bar = A_bar, B_bar
			newNcut = Ncut(A, B_bar, A_bar, B,sim,prefix_array)
			if newNcut<minNcut:
				minNcut = newNcut
				X,Y = A, B_bar
				X_bar, Y_bar = A_bar, B	
	depth+=1
	parent_id = node_id
	align_create_trees(X,Y,sim,prefix_array,sentence_tree,sent_id,depth,parent_id)
	align_create_trees(X_bar, Y_bar,sim,prefix_array,sentence_tree,sent_id,depth,parent_id)


def align_spanaoh_leaf_nonleaf(S,T, inner_seuil,sim,prefix_array,leaf_path,nonleaf_path):
	if len(S)==1 or len(T)==1:
			with open(leaf_path, 'a+') as file:
				file.write(f'{S[0]},{S[-1]}-{T[0]},{T[-1]} ')
			return S,T # terminate the procedure
	else:
		with open(nonleaf_path, 'a+') as file:
			file.write(f'{S[0]},{S[-1]}-{T[0]},{T[-1]} ')
		minNcut = 2
		maxNcut = 0.5
		X,Y = S,T
		# loop over the indices that are the potential cutting points
		for i in range(1,len(S)):
			for j in range(1,len(T)):
				A = S[:i]
				B = T[:j]
				A_bar = S[i:]
				B_bar = T[j:]
				newNcut = Ncut(A,B, A_bar, B_bar,sim,prefix_array)
				if newNcut<minNcut:
					minNcut = newNcut
					X,Y = A, B
					X_bar, Y_bar = A_bar, B_bar
				newNcut = Ncut(A, B_bar, A_bar, B,sim,prefix_array)
				if newNcut<minNcut:
					minNcut = newNcut
					X,Y = A, B_bar
					X_bar, Y_bar = A_bar, B	
		align_spanaoh_leaf_nonleaf(X,Y, inner_seuil,sim,prefix_array,leaf_path,nonleaf_path)
		align_spanaoh_leaf_nonleaf(X_bar, Y_bar,inner_seuil,sim,prefix_array,leaf_path,nonleaf_path)


if __name__=="__main__":
	all_trees = []
	# sentence_tree = Tree()
	# align_create_trees(source, target,sim,prefix_array,sentence_tree,depth=0,parent='',sent_id=num)
	# all_trees.append(sentence_tree)
	print([(len(tree),tree.depth()) for tree in all_trees])    # .show()
	# with open(leaf_path, 'a+') as leaf_file, open(nonleaf_path, 'a+') as nonleaf_file:
	# 	leaf_file.write(f'{num}\t')
	# 	nonleaf_file.write(f'{num}\t')
	# align_spanaoh_leaf_nonleaf(source, target, inner_seuil,sim,prefix_array,leaf_path,nonleaf_path)
	# with open(leaf_path, 'a+') as leaf_file, open(nonleaf_path, 'a+') as nonleaf_file:
	# 	leaf_file.write('\n')
	# 	nonleaf_file.write('\n')