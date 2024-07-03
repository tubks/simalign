from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Union
from collections import Counter

import numpy as np
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
try:
	import networkx as nx
	from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
except ImportError:
	nx = None
import torch
from transformers import BertModel, BertTokenizer, XLMModel, XLMTokenizer, RobertaModel, RobertaTokenizer, XLMRobertaModel, XLMRobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer


def extract_sentences(soup_or_path, is_path=False):
	if is_path:
		with open(soup_or_path, 'r') as filehandle:
			soup = BeautifulSoup(filehandle, 'xml')
	# the first linkList contains linkGroups with all the alignments (linkList level='chunk')
	linkGroups = soup.linkList.find_all('linkGroup')
	sentence_tuples = []
	for linkGroup in linkGroups:
		links = linkGroup.find_all('link')
		for link in links:
			if link['parentID'] == "ROOT":
				spans = link.find_all('docSpan')
				sentence_tuples.append(
					(spans[0].string, spans[1].string))
	return sentence_tuples

class EmbeddingLoader(object):
	def __init__(self, model: str="bert-base-multilingual-cased", device=torch.device('cpu'), layer: int=8):
		TR_Models = {
			'bert-base-uncased': (BertModel, BertTokenizer),
			'bert-base-multilingual-cased': (BertModel, BertTokenizer),
			'bert-base-multilingual-uncased': (BertModel, BertTokenizer),
			'xlm-mlm-100-1280': (XLMModel, XLMTokenizer),
			'roberta-base': (RobertaModel, RobertaTokenizer),
			'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer),
			'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer),
		}

		self.model = model
		self.device = device
		self.layer = layer
		self.emb_model = None
		self.tokenizer = None

		if model in TR_Models:
			model_class, tokenizer_class = TR_Models[model]
			self.emb_model = model_class.from_pretrained(model, output_hidden_states=True)
			self.emb_model.eval()
			self.emb_model.to(self.device)
			self.tokenizer = tokenizer_class.from_pretrained(model)
		else:
			# try to load model with auto-classes
			config = AutoConfig.from_pretrained(model, output_hidden_states=True)
			self.emb_model = AutoModel.from_pretrained(model, config=config)
			self.emb_model.eval()
			self.emb_model.to(self.device)
			self.tokenizer = AutoTokenizer.from_pretrained(model)

	def get_embed_list(self, sent_batch: List[List[str]]) -> torch.Tensor:
		if self.emb_model is not None:
			with torch.no_grad():
				if not isinstance(sent_batch[0], str):
					inputs = self.tokenizer(sent_batch, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
				else:
					inputs = self.tokenizer(sent_batch, is_split_into_words=False, padding=True, truncation=True, return_tensors="pt")
				hidden = self.emb_model(**inputs.to(self.device))["hidden_states"]
				if self.layer >= len(hidden):
					raise ValueError(f"Specified to take embeddings from layer {self.layer}, but model has only {len(hidden)} layers.")
				outputs = hidden[self.layer]
				return outputs[:, 1:-1, :]
		else:
			return None


class SentenceAligner(object):
	def __init__(self, model: str = "bert", token_type: str = "bpe", distortion: float = 0.0, matching_methods: str = "mai", device: str = "cpu", layer: int = 8):
		model_names = {
			"bert": "bert-base-multilingual-cased",
			"xlmr": "xlm-roberta-base"
			}
		all_matching_methods = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}

		self.model = model
		if model in model_names:
			self.model = model_names[model]
		self.token_type = token_type
		self.distortion = distortion
		self.matching_methods = [all_matching_methods[m] for m in matching_methods]
		self.device = torch.device(device)

		self.embed_loader = EmbeddingLoader(model=self.model, device=self.device, layer=layer)

	@staticmethod
	def get_max_weight_match(sim: np.ndarray) -> np.ndarray:
		if nx is None:
			raise ValueError("networkx must be installed to use match algorithm.")
		def permute(edge):
			if edge[0] < sim.shape[0]:
				return edge[0], edge[1] - sim.shape[0]
			else:
				return edge[1], edge[0] - sim.shape[0]
		G = from_biadjacency_matrix(csr_matrix(sim))
		matching = nx.max_weight_matching(G, maxcardinality=True)
		matching = [permute(x) for x in matching]
		matching = sorted(matching, key=lambda x: x[0])
		res_matrix = np.zeros_like(sim)
		for edge in matching:
			res_matrix[edge[0], edge[1]] = 1
		return res_matrix

	@staticmethod
	def get_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
		return (cosine_similarity(X, Y) + 1.0) / 2.0
	
	@staticmethod
	def get_similarity_cos_squared(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
		return cosine_similarity(X,Y)**4
	
	@staticmethod
	def get_similarity_normalized_squared(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
		return ((cosine_similarity(X, Y) + 1.0) / 2.0)**2
	
	@staticmethod
	def get_similarity_cosine(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
		return cosine_similarity(X, Y)
	
	@staticmethod
	def average_embeds_over_words(bpe_vectors: np.ndarray, word_tokens_pair: List[List[str]]) -> List[np.array]:
		w2b_map = []
		cnt = 0
		w2b_map.append([])
		for wlist in word_tokens_pair[0]:
			w2b_map[0].append([])
			for x in wlist:
				w2b_map[0][-1].append(cnt)
				cnt += 1
		cnt = 0
		w2b_map.append([])
		for wlist in word_tokens_pair[1]:
			w2b_map[1].append([])
			for x in wlist:
				w2b_map[1][-1].append(cnt)
				cnt += 1

		new_vectors = []
		for l_id in range(2):
			w_vector = []
			for word_set in w2b_map[l_id]:
				w_vector.append(bpe_vectors[l_id][word_set].mean(0))
			new_vectors.append(np.array(w_vector))
		return new_vectors

	@staticmethod
	def get_alignment_matrix(sim_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		m, n = sim_matrix.shape
		forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
		backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
		return forward, backward.transpose()

	@staticmethod
	def apply_distortion(sim_matrix: np.ndarray, ratio: float = 0.5) -> np.ndarray:
		shape = sim_matrix.shape
		if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
			return sim_matrix

		pos_x = np.array([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])])
		pos_y = np.array([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])])
		distortion_mask = 1.0 - ((pos_x - np.transpose(pos_y)) ** 2) * ratio

		return np.multiply(sim_matrix, distortion_mask)

	@staticmethod
	def iter_max(sim_matrix: np.ndarray, max_count: int=2) -> np.ndarray:
		alpha_ratio = 0.9
		m, n = sim_matrix.shape
		forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
		backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
		inter = forward * backward.transpose()

		if min(m, n) <= 2:
			return inter

		new_inter = np.zeros((m, n))
		count = 1
		while count < max_count:
			mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
			mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
			mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
			mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
			if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
				mask *= 0.0
				mask_zeros *= 0.0

			new_sim = sim_matrix * mask
			fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
			bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
			new_inter = fwd * bac

			if np.array_equal(inter + new_inter, inter):
				break
			inter = inter + new_inter
			count += 1
		return inter

	def get_word_aligns(self, src_sent: Union[str, List[str]], trg_sent: Union[str, List[str]]) -> Dict[str, List]:
		if isinstance(src_sent, str):
			src_sent = src_sent.split()
		if isinstance(trg_sent, str):
			trg_sent = trg_sent.split()
		l1_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in src_sent]
		l2_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in trg_sent]
		bpe_lists = [[bpe for w in sent for bpe in w] for sent in [l1_tokens, l2_tokens]]

		if self.token_type == "bpe":
			l1_b2w_map = []
			for i, wlist in enumerate(l1_tokens):
				l1_b2w_map += [i for x in wlist]
			l2_b2w_map = []
			for i, wlist in enumerate(l2_tokens):
				l2_b2w_map += [i for x in wlist]

		vectors = self.embed_loader.get_embed_list([src_sent, trg_sent]).cpu().detach().numpy()
		vectors = [vectors[i, :len(bpe_lists[i])] for i in [0, 1]]

		if self.token_type == "word":
			vectors = self.average_embeds_over_words(vectors, [l1_tokens, l2_tokens])

		all_mats = {}
		sim = self.get_similarity(vectors[0], vectors[1])
		print(f"SIM:{sim}")
		sim = self.apply_distortion(sim, self.distortion)

		all_mats["fwd"], all_mats["rev"] = self.get_alignment_matrix(sim)
		all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]
		if "mwmf" in self.matching_methods:
			all_mats["mwmf"] = self.get_max_weight_match(sim)
		if "itermax" in self.matching_methods:
			all_mats["itermax"] = self.iter_max(sim)

		aligns = {x: set() for x in self.matching_methods}
		for i in range(len(vectors[0])):
			for j in range(len(vectors[1])):
				for ext in self.matching_methods:
					if all_mats[ext][i, j] > 0:
						if self.token_type == "bpe":
							aligns[ext].add((l1_b2w_map[i], l2_b2w_map[j]))
						else:
							aligns[ext].add((i, j))
		for ext in aligns:
			aligns[ext] = sorted(aligns[ext])
		return aligns

	def get_similarity_matrix(self, src_sent, trg_sent):
		if isinstance(src_sent, str):
			src_sent = src_sent.split()
		if isinstance(trg_sent, str):
			trg_sent = trg_sent.split()
		l1_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in src_sent]
		l2_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in trg_sent]
		bpe_lists = [[bpe for w in sent for bpe in w] for sent in [l1_tokens, l2_tokens]]

		if self.token_type == "bpe":
			l1_b2w_map = []
			for i, wlist in enumerate(l1_tokens):
				l1_b2w_map += [i for x in wlist]
			l2_b2w_map = []
			for i, wlist in enumerate(l2_tokens):
				l2_b2w_map += [i for x in wlist]

		vectors = self.embed_loader.get_embed_list([src_sent, trg_sent]).cpu().detach().numpy()
		vectors = [vectors[i, :len(bpe_lists[i])] for i in [0, 1]]

		if self.token_type == "word":
			vectors = self.average_embeds_over_words(vectors, [l1_tokens, l2_tokens])

		sim = self.get_similarity_cos_squared(vectors[0], vectors[1])
		return sim

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
		# print("PREFIX INDICES: ",row_start,row_end, col_start, col_end)
		# print("PREFIX SUMS: ",prefix_array[row_end-1,col_end-1])
		# prefix_slice = prefix_array[row_start:row_end-1, col_start:col_end-1]
		# print("PREFIX SLICE: ", prefix_slice)
		# print("PREFIX SLICE SHAPE: ", prefix_slice.shape)
		return prefix_array[row_end-1,col_end-1]
	if row_start==0 and col_start!=0:
		# print("PREFIX INDICES: ",row_start,row_end, col_start, col_end)
		# print("PREFIX SUMS: ", prefix_array[row_end-1,col_end-1]-prefix_array[row_end-1,col_start-1])
		# prefix_slice = prefix_array[row_start-1:row_end-1, col_start-1:col_end-1]
		# print("PREFIX SLICE: ", prefix_slice)
		# print("PREFIX SLICE SHAPE: ", prefix_slice.shape)
		return prefix_array[row_end-1,col_end-1]-prefix_array[row_end-1,col_start-1]
	if row_start!=0 and col_start==0:
		# print("PREFIX INDICES: ",row_start,row_end, col_start, col_end)
		# print("PREFIX SUMS: ",prefix_array[row_end-1,col_end-1]-prefix_array[row_start-1,col_end-1])
		# prefix_slice = prefix_array[row_start-1:row_end-1, col_start:col_end-1]
		# print("PREFIX SLICE: ", prefix_slice)
		# print("PREFIX SLICE SHAPE: ", prefix_slice.shape)
		return prefix_array[row_end-1,col_end-1]-prefix_array[row_start-1,col_end-1]
	else:
		# print("PREFIX INDICES: ",row_start,row_end, col_start, col_end)
		# print("PREFIX SUMS: ",prefix_array[row_end-1, col_end-1], prefix_array[row_start-1,col_start-1], prefix_array[row_end-1,col_end-1]-prefix_array[row_end-1,col_start-1]-prefix_array[row_start-1,col_end-1]+prefix_array[row_start-1,col_start-1])
		# prefix_slice = prefix_array[row_start-1:row_end-1, col_start-1:col_end-1]
		# print("PREFIX SLICE: ", prefix_slice)
		# print("PREFIX SLICE SHAPE: ", prefix_slice.shape)
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
		# print("SIM SUM: ",np.sum(sim[X[0]:X[-1],Y[0]:Y[-1]]))
		# print("SIM SLICE SHAPE: ", sim[X[0]:X[-1],Y[0]:Y[-1]].shape)
		return compute_score_from_prefix_array(X[0],X[-1],Y[0],Y[-1])
		# return np.sum(sim[X[0]:X[-1],Y[0]:Y[-1]])
def cut(X,Y, X_bar, Y_bar):
	# print("cut:",X,Y,X_bar,Y_bar)
	return W(X, Y_bar)+W(X_bar, Y)

def Ncut(X, Y, X_bar, Y_bar):
	return cut(X,Y,X_bar, Y_bar)/(cut(X,Y,X_bar, Y_bar)+2*W(X,Y))+cut(X_bar, Y_bar, X, Y)/(cut(X_bar, Y_bar, X, Y)+2*W(X_bar, Y_bar))

def align(S,T):
	# print("START ",S, T)
	if len(S)==1 or len(T)==1:
		fr_lens.append(len(S))
		eng_lens.append(len(T))
		for word_s in S:
			for word_t in T:
				with open(path_to_save, 'a+') as file:
					# print("END", word_s, word_t)
					file.write(f'{word_s}-{word_t} ')    # should the cases containing more than 1 word be saved as possible links?
		return S,T # terminate the procedure
	
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
			# print(A,B,A_bar,B_bar)
			# print(Ncut(A,B, A_bar, B_bar))
			# print(Ncut(A, B_bar, A_bar, B))
			newNcut = Ncut(A,B, A_bar, B_bar)
			# if newNcut>maxNcut:
				# maxNcut=newNcut
				# maxCuts.append(maxNcut)
			if newNcut<minNcut:
				minNcut = newNcut
				# print("minNcut update", minNcut)
				# minCuts.append(minNcut)
				X,Y = A, B
				X_bar, Y_bar = A_bar, B_bar
			newNcut = Ncut(A, B_bar, A_bar, B)
			# if newNcut>maxNcut:
			# 	maxNcut=newNcut
			# 	maxCuts.append(maxNcut)
			if newNcut<minNcut:
				minNcut = newNcut
				# print("minNcut update", minNcut)
				# minCuts.append(minNcut)
				X,Y = A, B_bar
				X_bar, Y_bar = A_bar, B
			
	align(X,Y)
	align(X_bar, Y_bar)

ali_xml_paths = ["dat/xml_ali/LAuberge_TheInn.ali.xml", "dat/xml_ali/BarbeBleue_BlueBeard.ali.xml", "dat/xml_ali/Laderniereclasse_Thelastlesson.ali.xml", "dat/xml_ali/LaVision_TheVision.ali.xml"]
path = "dat/xml_ali/ChatBotte_MasterCat.ali.xml"

model = SentenceAligner(token_type='word')   # simalign class
for i,path in enumerate(ali_xml_paths):
	path_to_save = f'{i}_puissance_4.txt'
	sentence_tuples = extract_sentences(path, is_path=True)
	# minCuts = []
	# maxCuts = []
	fr_lens = []
	eng_lens = []
	for num, tup in enumerate(sentence_tuples):
		source_sentence, target_sentence = tup
		# print(source_sentence, target_sentence)
		print(num)
		sim = model.get_similarity_matrix(source_sentence, target_sentence)
		# print(sim)
		prefix_array = build_prefix_array(sim)
		source = [i for i in range(len(source_sentence.split()))] # list containing word indices
		target = [i for i in range(len(target_sentence.split()))]
		#print(source, target)
		with open(path_to_save, 'a+') as file:
			file.write(f'{num}\t')
		align(source, target)
		with open(path_to_save, 'a+') as file:
			file.write('\n')
	# minCuts.sort()
	# print(maxCuts, minCuts)
	# print(len(minCuts), minCuts[:10])
	# maxCuts.sort(reverse=True)
	# print(len(maxCuts), maxCuts[:10])
	# counter_fr = Counter(fr_lens)
	# counter_eng = Counter(eng_lens)
	# print("FR: ",counter_fr,"ENG: ", counter_eng)