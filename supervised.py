import codecs
import os
import sys
import time


tags = ['NN', 'NST', 'NNP', 'PRP', 'DEM', 'VM', 'VAUX', 'JJ', 'RB', 'PSP', 'RP', 'CC', 'WQ', 'QF', 'QC', 'QO', 'CL', 'INTF', 'INJ', 'NEG', 'UT', 'SYM', 'COMP', 'RDP', 'ECH', 'UNK', 'XC']

tags_desc = {
	'NN': 'Noun (common or proper)',
	'NST': 'Noun (pronoun)',
	'NNP': 'Noun (proper noun)',
	'PRP': 'Pronoun (personal pronoun)',
	'DEM': 'Determiner (demonstrative determiner)',
	'VM': 'Verb (main verb)',
	'VAUX': 'Verb (auxiliary verb)',
	'JJ': 'Adjective (adjective)',
	'RB': 'Adverb (adverb)',
	'PSP': 'Postposition (postposition)',
	'RP': 'Particle (particle)',
	'CC': 'Conjunction (coordinating conjunction)',
	'WQ': 'Wh-question (wh-question)',
	'QF': 'Quantifier (quantifier)',
	'QC': 'Cardinal number (cardinal number)',
	'QO': 'Ordinal number (ordinal number)',
	'CL': 'Classifier (classifier)',
	'INTF': 'Interjection (interjection)',
	'INJ': 'Injection (injection)',
	'NEG': 'Negative marker (negative marker)',
	'UT': 'Foreign word (foreign word)',
	'SYM': 'Symbol (symbol)',
	'COMP': 'Comparative marker (comparative marker)',
	'RDP': 'Reduplication (reduplication)',
	'ECH': 'Echo word (echo word)',
	'UNK': 'Unknown word (unknown word)',
	'XC': 'Other (other)'
}
#--------------------------------------------------------------------------
# Function: max_connect
#--------------------------------------------------------------------------
# 	Description
#		
#	max_connect function performs the viterbi decoding. Choosing which tag
#	for the current word leads to a better tag sequence. 
#
#--------------------------------------------------------------------------
def max_connect(x, y, viterbi_matrix, emission, transmission_matrix):
	max = -99999
	path = -1
	
	for k in range(len(tags)):
		val = viterbi_matrix[k][x-1] * transmission_matrix[k][y]
		if val * emission > max:
			max = val
			path = k
	return max, path

#--------------------------------------------------------------------------
# 	Description
#		
#	1) Unique words are extracted from the training data.
#	2) Count of occurence of each tag is calculated.
#	3) Emission & Transmission matrix are initialized and computed.
#	4) Testing data is read.
#	5) Trellis for viterbi decoding is computed.
#	6) Path is printed on to output file.
#
#--------------------------------------------------------------------------
def main():

	filepath = "./data/hindi_training.txt" 
	languages = "hindi"
	wordtypes = []
	tagscount = []
	exclude = ["<s>", "</s>", "START", "END"]
	f = codecs.open(filepath, 'r', encoding='utf-8')
	file_contents = f.readlines()

	# Initialize count of each tag to Zero's
	for x in range(len(tags)):
		tagscount.append(0)

	# Calculate count of each tag in the training corpus and also the wordtypes in the corpus
	for x in range(len(file_contents)):
		line = file_contents.pop(0).strip().split(' ')
		for i, word in enumerate(line):
			if i == 0:
				if word not in wordtypes and word not in exclude:
					wordtypes.append(word)
			else:
				if word in tags and word not in exclude:
					tagscount[tags.index(word)] += 1
	f.close()
	
	# Declare variables for emission and transmission matrix	
	emission_matrix = []
	transmission_matrix = []
		
	# Initialize emission matrix
	for x in range(len(tags)):
		emission_matrix.append([])
		for y in range(len(wordtypes)):
			emission_matrix[x].append(0)

	# Initialize transmission matrix
	for x in range(len(tags)):
		transmission_matrix.append([])
		for y in range(len(tags)):
			transmission_matrix[x].append(0)

	# Open training file to update emission and transmission matrix
	f = codecs.open(filepath, 'r', encoding='utf-8')
	file_contents = f.readlines()

	# Update emission and transmission matrix with appropriate counts
	row_id = -1
	for x in range(len(file_contents)):
		line = file_contents.pop(0).strip().split(' ')

		if line[0] not in exclude:
			col_id = wordtypes.index(line[0])
			prev_row_id = row_id
			row_id = tags.index(line[1])
			emission_matrix[row_id][col_id] += 1
			if prev_row_id != -1:
				transmission_matrix[prev_row_id][row_id] += 1
		else:
			row_id = -1

	
	# Divide each entry in emission matrix by appropriate tag count to store probabilities in each entry instead of just count
	for x in range(len(tags)):
		for y in range(len(wordtypes)):
			if tagscount[x] != 0:
				emission_matrix[x][y] = float(emission_matrix[x][y]) / tagscount[x]

	# Divide each entry in transmission matrix by appropriate tag count to store probabilities in each entry instead of just count
	for x in range(len(tags)):
		for y in range(len(tags)):
			if tagscount[x] != 0:
				transmission_matrix[x][y] = float(transmission_matrix[x][y]) / tagscount[x]

	#   TESTING Starts here

	hindi_text = "धार जिला ऐतिहासिक और सांस्कृतिक रूप से महत्वपूर्ण स्थान रखता है"

	# Open the testing file to read test sentences
	test_input = hindi_text.split(' ')
	
	# Declare variables for test words and pos tags
	test_words = []
	pos_tags = []

	# For each line POS tags are computed
	for j in range(len(test_input)):
		
		test_words = []
		pos_tags = []

		line = test_input.pop(0).strip().split(' ')
		
		for word in line:
			test_words.append(word)
			pos_tags.append(-1)

		viterbi_matrix = []
		viterbi_path = []
		
		# Initialize viterbi matrix of size |tags| * |no of words in test sentence|
		for x in range(len(tags)):
			viterbi_matrix.append([])
			viterbi_path.append([])
			for y in range(len(test_words)):
				viterbi_matrix[x].append(0)
				viterbi_path[x].append(0)

		# Update viterbi matrix column wise
		for x in range(len(test_words)):
			for y in range(len(tags)):
				if test_words[x] in wordtypes:
					word_index = wordtypes.index(test_words[x])
					tag_index = tags.index(tags[y])
					emission = emission_matrix[tag_index][word_index]
				else:
					emission = 0.001

				if x > 0:
					max, viterbi_path[y][x] = max_connect(x, y, viterbi_matrix, emission, transmission_matrix)
				else:
					max = 1
				viterbi_matrix[y][x] = emission * max

		# Identify the max probability in last column i.e. best tag for last word in test sentence
		maxval = -999999
		maxs = -1
		for x in range(len(tags)):
			if viterbi_matrix[x][len(test_words)-1] > maxval:
				maxval = viterbi_matrix[x][len(test_words)-1]
				maxs = x
			
		# Backtrack and identify best tags for each words
		for x in range(len(test_words)-1, -1, -1):
			pos_tags[x] = maxs
			maxs = viterbi_path[maxs][x]

		# Print output to the file.	
		for i, x in enumerate(pos_tags):
			print(test_words[i] + " ===> " + tags_desc[tags[x]])




if __name__ == "__main__":
	main()