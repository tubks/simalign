from calc_align_score import load_gold, calc_score
from divisive_alignment import align, build_prefix_array, extract_sentences, SentenceAligner
import numpy as np


if __name__=="__main__":
	path = "dat/xml_ali/ChatBotte_MasterCat.ali.xml"
	ref_ali = "dat/gold_ali_from_xml/ali_w2w_ChatBotte_MasterCat.txt"
	path_to_save = "alarechercheduchat.txt"
	result_path = 'hyperparam_scores.txt'
	probs,surs,surs_count = load_gold(ref_ali)
	model = SentenceAligner(token_type='word')   # simalign class	# model="xlmr",
	k=4
	sent_tuples = extract_sentences(path, is_path=True)
	for inner_seuil in range(61,70, 1):
		inner_seuil = inner_seuil/1000.0
		with open(path_to_save, 'w') as file:
				file.write('')
		for num, tup in enumerate(sent_tuples):
			source_sentence, target_sentence = tup
			vec1,vec2 = model.get_embeddings(source_sentence, target_sentence)
			mean_cos_fr, mean_cos_eng = model.get_mean_similarity_to_neighbs(vec1,vec2, k=k)
			sim = model.get_csls(vec1,vec2, k,mean_cos_fr, mean_cos_eng)
			prefix_array = build_prefix_array(sim)
			source = [i for i in range(len(source_sentence.split()))] # list containing word indices
			target = [i for i in range(len(target_sentence.split()))]
			with open(path_to_save, 'a+') as file:
				file.write(f'{num}\t')
			align(source, target, inner_seuil,sim,prefix_array,path_to_save)
			with open(path_to_save, 'a+') as file:
				file.write('\n')
		y_prec, y_rec, y_f1, aer = calc_score(path_to_save,probs,surs,surs_count)
		with open(result_path, 'a+') as file:
			file.write("norm\tseuil: {}\tPrec: {}\tRec: {}\tF1: {}\tAER: {}\n".format(inner_seuil,y_prec, y_rec, y_f1, aer))
		print(inner_seuil)