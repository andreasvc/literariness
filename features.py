# -*- coding: UTF-8 -*-
"""Extract stylometric features from parsed texts.

Usage:

    python3 features.py <dataset> <target>

Where `<dataset>` is a directory containing texts and metadata.

Input
-----
- texts with filenames of the form:

    dataset/parses/Author_Title.mrg
    OR
    dataset/parses/Author_Title.dbr

    Where Author_Title.mrg is a file with regular bracketed trees, one per
    line; alternatively, Author_Title.dbr is a file with discontinuous
    bracketed trees,
    cf. http://discodop.readthedocs.io/en/latest/fileformats.html#discbracket
- a file 'dataset/metadata.csv' describing the texts; specificallly,
    a column named "Label" should match the filenames of the texts
	(minus the treebank format extension), and the column <target> is a column
	that is used as the target for the regression/classification task.

Output
------
- Written to features/ (must not exist) in a series of compressed CSV files.
    Extracted features with prefixes::

        .: punctuation
        r: readability scores
        b: assorted more or less basic features
        w: word unigrams
        c: character n-grams
        s: syntactic categories
        t: POS tags
        p: grammar productions
        q: pq-grams
        F: fragments from external grammar
        f: fragments mined from training folds

Options
-------
--slice=n:m         select which sentence numbers to use from each file,
                    in Python slice notation; defaults to `0:1000`.
--numproc=n         the number of proccesses to start to exploit multiple
                    cores; defaults to `1`.
--folds=n           if metadata.csv does not already contain a `folds`
                    column, the dataset will be automatically divided into
                    `n` folds, stratified according to the target variable.
--disc              flag that must be given when parse trees contain
                    discontinuous constituents; affects the representation
                    of grammar productions and tree fragments.
--lang=<en|nl>      set language; used to select stop word list, POS tags
                    of function words, and syllabification for the
                    readability measures. only affects readability and
                    style n-gram features.
--freqlist=<filename>
                    optionally, a filename with a tab-separated list of
                    word frequencies from an external corpus (lower cased).
--extfrags=<filename>
                    optionally, a filename with predefined tree fragments
                    to count in the texts and use as features.
--nofragments       disables fragment mining

Fragment mining thresholds:

--nonzerocount=x    minimum proportion [0, 1] of nonzero counts for a fragment,
                    used as initial filter. `None` to disable.
--minfreq=x         minimum overall count (in all texts combined) of a feature.
                    [0, ...]; `None` to disable.
--relevancy=x       p-value cutoff [0, 1] for regression/chi2 test of fragment
                    counts versus target values.
--redundancy=x      correlation (r) or symmetric uncertainty threshold [0, 1]
					beyond which features are considered redundant."""

from __future__ import division, print_function
import io
import os
try:
	import re2 as re
except ImportError:
	import re
import sys
import glob
import gzip
import bz2
import lzma
from itertools import islice
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from getopt import gnu_getopt, GetoptError
import faulthandler
import numpy as np
import pandas
from scipy.stats import entropy
import scipy.sparse
from sklearn import cross_validation, feature_selection
from sklearn.utils import check_random_state
from nltk import ngrams
from discodop.tree import Tree, ParentedTree, ptbunescape
from discodop.heads import applyheadrules, readheadrules
from discodop.grammar import lcfrsproductions, printrule
from discodop.plcfrs import merge
from discodop.containers import Vocabulary, FixedVocabulary, Ctrees
from discodop.treetransforms import binarize, unbinarize, handledisc
from discodop import treebank, treebanktransforms, treesearch, _fragments, \
		punctuation
import readability

DIRECTSPEECHRE = re.compile(r"^- .*$|(?:^|.* )['\"](?: .*|$)")
STOPWORDS = {
		'nl': set((
		' aan af al alles als altijd andere ben bij daar dan dat'
		' de der deze die dit doch doen door dus een eens en er ge'
		' geen geweest haar had heb hebben heeft hem het hier hij'
		' hoe hun iemand iets ik in is ja je kan kon kunnen maar'
		' me meer men met mij mijn moet na naar niet niets nog nu'
		' of om omdat ons ook op over reeds te tegen toch toen tot'
		' u uit uw van veel voor want waren was wat we wel werd'
		' wezen wie wij wil worden zal ze zei zelf zich zij zijn'
		' zo zonder zou').split()),
		'en': set("""
		's 'll a a's able about above according accordingly across actually
		after afterwards again against ain't all allow allows almost alone
		along already also although always am among amongst an and another any
		anybody anyhow anyone anything anyway anyways anywhere apart appear
		appreciate appropriate are aren't around as aside ask asking associated
		at available away awfully b be became because become becomes becoming
		been before beforehand behind being believe below beside besides best
		better between beyond both brief but by c c'mon c's came can can't
		cannot cant cause causes certain certainly changes clearly co com come
		comes concerning consequently consider considering contain containing
		contains corresponding could couldn't course currently d definitely
		described despite did didn't different do does doesn't doing don't done
		down downwards during e each edu eg eight either else elsewhere enough
		entirely especially et etc even ever every everybody everyone
		everything everywhere ex exactly example except f far few fifth first
		five followed following follows for former formerly forth four from
		further furthermore g get gets getting given gives go goes going gone
		got gotten greetings h had hadn't happens hardly has hasn't have
		haven't having he he's hello help hence her here here's hereafter
		hereby herein hereupon hers herself hi him himself his hither hopefully
		how howbeit however i i'd i'll i'm i've ie if ignored immediate in
		inasmuch inc indeed indicate indicated indicates inner insofar instead
		into inward is isn't it it'd it'll it's its itself j just k keep keeps
		kept know knows known l last lately later latter latterly least less
		lest let let's like liked likely little look looking looks ltd m mainly
		many may maybe me mean meanwhile merely might more moreover most mostly
		mr mr. much must my myself n n't name namely nd near nearly necessary
		need needs neither never nevertheless new next nine no nobody non none
		noone nor normally not nothing novel now nowhere o obviously of off
		often oh ok okay old on once one ones only onto or other others
		otherwise ought our ours ourselves out outside over overall own p
		particular particularly per perhaps placed please plus possible
		presumably probably provides q que quite qv r rather rd re really
		reasonably regarding regardless regards relatively respectively right s
		said same saw say saying says second secondly see seeing seem seemed
		seeming seems seen self selves sensible sent serious seriously seven
		several shall she should shouldn't since six so some somebody somehow
		someone something sometime sometimes somewhat somewhere soon sorry
		specified specify specifying still sub such sup sure t t's take taken
		tell tends th than thank thanks thanx that that's thats the their
		theirs them themselves then thence there there's thereafter thereby
		therefore therein theres thereupon these they they'd they'll they're
		they've think third this thorough thoroughly those though three through
		throughout thru thus to together too took toward towards tried tries
		truly try trying twice two u un under unfortunately unless unlikely
		until unto up upon us use used useful uses using usually uucp v value
		various very via viz vs w want wants was wasn't way we we'd we'll we're
		we've welcome well went were weren't what what's whatever when whence
		whenever where where's whereafter whereas whereby wherein whereupon
		wherever whether which while whither who who's whoever whole whom whose
		why will willing wish with within without won't wonder would would
		wouldn't x y yes yet you you'd you'll you're you've your yours yourself
		yourselves z zero""".split()),
		}


class Text(object):
	"""Load a parsed text and extract various features from it.

	:param maxlen: filter out incorrectly segmented sentences.
	"""
	@classmethod
	def fromparsetrees(cls, filename, start=None, end=None, disc=False,
			maxlen=300, lang='en'):
		self = cls()
		self.filename = filename
		self.lang = lang
		self.disc = disc
		if filename.endswith('.mrg'):
			corpus = treebank.BracketCorpusReader(filename)
		elif filename.endswith('.dbr'):
			corpus = treebank.DiscBracketCorpusReader(filename)
		elif filename.endswith('.export'):
			corpus = treebank.NegraCorpusReader(filename)
		else:
			raise ValueError('pass filename with .mrg, .dbr, '
					'or .export extension.')
		selection = [item for _, item
				in corpus.itertrees(start, end)
				if len(item.sent) <= maxlen]
		if not selection:
			raise ValueError('no trees in interval %d:%d <= %d words' % (
					start, end, maxlen))
		self.trees = [item.tree for item in selection]
		self.origsents = [item.sent for item in selection]
		self.sents = [list(map(ptbunescape, sent)) for sent in self.origsents]
		self.detokenized = [detokenize(' '.join(sent)) for sent in self.sents]
		self.start, self.end = start, end

		# NB: copy the relevant files to the dataset directory from
		# http://github.com/andreasvc/disco-dop/
		headrules = {
				'en': 'ptb.headrules',
				'nl': 'alpino.headrules',
				'de': 'negra.headrules',
				'fr': 'ftb.headrules',
				}.get(lang, None)
		if not headrules or not os.path.exists(headrules):
			headrules = None
		headrules = readheadrules(headrules) if headrules else {}
		self.unbinarized = []
		self.sentsnopunct = []
		for tree, sent in zip(self.trees, self.sents):
			tree = unbinarize(tree.copy(True))
			if lang == 'nl':
				tree = treebanktransforms.reversetransform(tree,
						treebanktransforms.PRESETS['lassy-func'])
			sent = sent.copy()
			punctuation.punctremove(tree, sent)
			applyheadrules(tree, headrules)
			if tree:
				self.unbinarized.append(tree)
				self.sentsnopunct.append(sent)

	@classmethod
	def fromtokenized(cls, filename, tokens, start=None, end=None, maxlen=300,
			lang='en'):
		"""Alternate constructor when only a tokenized text is available.

		:param tokens: a multi-line string of space-separated tokens."""
		self = cls()
		self.filename = filename
		self.lang = lang
		self.disc = False  # not applicable
		self.start, self.end = start, end
		tok = (line.split() for line in islice(tokens.splitlines(), start, end))
		selection = [sent for sent in tok if len(sent) <= maxlen]
		if not selection:
			raise ValueError('no sentences in interval %d:%d <= %d words' % (
					start, end, maxlen))
		self.sents = self.origsents = [item for item in selection]
		self.detokenized = [detokenize(' '.join(sent)) for sent in self.sents]
		self.trees = self.unbinarized = self.sentsnopunct = []
		return self

	def readabilityscores(self):
		"""Return simple readability measures for text."""
		result = readability.getmeasures(
				(' '.join(sent) for sent in self.sents),
				lang=self.lang)
		return {'r_' + a: b
				for data in result.values()
					for a, b in data.items()}

	def basicfeatures(self, freqlist=None):
		"""Compute several basic textual features.

		:param freqlist: if given, a DataFrame with word frequencies."""
		result = {}
		result['b_%_direct_speech'] = 100 * sum(1 for sent in self.sents
					if DIRECTSPEECHRE.search(' '.join(sent)) is not None
					) / len(self.sents)
		if freqlist is not None:
			thresholds = [1000, 2000, 3000, 5000, 10000, 20000]
			w = OrderedDict([('b_top{}vocab'.format(n), freqlist.ix[:n].index)
				for n in thresholds])
			# original: lines 1000 - 2000, including empty lines
			text = pandas.Series(Counter((
					token.lower() for sent in self.sents
					for token in sent)))
			if len(text) > 0:
				for a, b in w.items():
					result[a] = text.ix[text.index & b].sum() / text.sum()
				result['b_unknownvocab'] = (text.ix[text.index.difference(
						freqlist.index)].sum() / text.sum())
		# original (where filename is .tok file):
		# with open(self.filename, 'rb') as inp:
		# 	data = inp.read(100000)
		data = '\n'.join(' '.join(sent) for sent in self.sents
				).encode('utf8')[:30000]
		result['b_gzip_ratio'] = len(gzip.compress(data)) / float(len(data))
		result['b_bzip2_ratio'] = len(bz2.compress(data)) / float(len(data))
		result['b_lzma_ratio'] = (len(lzma.compress(data, preset=9))
				/ float(len(data)))
		totlen = numdeps = numsents = totdlevel = 0
		# NB: the following 2 features require unbinarized parse trees so that
		# heads can be identified; D-level additionally requires
		# functions/morph. feats and is only implemented for Dutch.
		for tree in self.unbinarized:
			if tree and self.lang == 'nl':
				totdlevel += treebanktransforms.dlevel(tree)
				numsents += 1
			if tree:
				a, b = treebank.deplen(treebank.dependencies(tree))
				totlen += a
				numdeps += b
		if numdeps:
			result['b_avgdeplen'] = totlen / numdeps
		if self.lang == 'nl' and numsents:
			result['b_avgdlevel'] = totdlevel / numsents
		return result

	def punctfrequencies(self):
		"""Return frequencies of non-alphanumeric characters for the text."""
		return Counter('._' + word
				for sent in self.sents
					for word in sent if not word[0].isalpha())

	def wordngrams(self, n):
		"""Return frequencies of word n-grams in text."""
		return Counter('w_' + '_'.join(ngram)
				for sent in self.sents
					for ngram in ngrams(sent, n))

	def stylengrams(self, n):
		"""Return frequencies of style n-grams in text.

		A style n-gram is a word n-gram in which content words are replaced
		with their POS tags."""
		if self.lang == 'en':
			funcwordpos = re.compile(
					r'^(?:NN(?:[PS]|PS)?|(?:JJ|RB)[RS]?|VB[DGNPZ])$')
			stopwords = STOPWORDS['en']
		elif self.lang == 'nl':
			# POS = 'let|spec|vg|lid|vnw|tw|tsw|vz|bw|ww|adj|n'
			# function words missing after applying these POS:
			# some adverbs, aux verbs.
			funcwordpos = re.compile(
					'^(?:'
					'let'   # punctuation
					'|vg'   # conjuction
					'|lid'  # determiner
					'|vnw'  # pronouns
					'|tsw'  # numeral
					'|vz'   # preposition
					')(?:$|/)')
			stopwords = STOPWORDS['nl']
		else:
			funcwordpos = re.compile('<dummy>')
			stopwords = set()
		return Counter('S_' + '_'.join(ngram)
				for tree, sent in zip(self.trees, self.sents)
					for ngram in ngrams(
						[word if funcwordpos.match(pos)
							or word in stopwords else pos.split('/')[0].upper()
							for word, (_, pos) in zip(sent, tree.pos())], n))

	def characterngrams(self, n):
		"""Return frequencies of character n-grams in text."""
		return Counter('c_' + ''.join(ngram)
				for sent in self.detokenized
					for ngram in ngrams(sent, n))

	def postags(self):
		"""Return counts of all POS tags in trees."""
		return Counter('t_' + pos
				for tree in self.trees
					for _word, pos in tree.pos())

	def syntacticcategories(self):
		"""Return counts of all syntactic categories in trees."""
		return Counter('s_' + node.label
				for tree in self.unbinarized
					for node in tree.subtrees()
					if node and isinstance(node[0], Tree))

	def productions(self, parents=False, lexical=False):
		"""Return counts of all grammar productions in trees.

		:param parents: if True, include parent on LHS.
		:param lexical: if True, include lexical productions POS => word."""
		if self.disc:
			return Counter('p_' + printrule(rule, yf)
					for tree, sent in zip(self.unbinarized, self.sentsnopunct)
						for rule, yf in lcfrsproductions(
							addparents(tree) if parents else tree,
							sent)
						if lexical or rule[1] != 'Epsilon')
		else:
			return Counter('p_' + prod
					for tree, sent in zip(self.unbinarized, self.sentsnopunct)
						for prod in cfgproductions(
							tree, sent, parents=parents, lexical=lexical))

	def pqgrams(self, p, q):  # pylint: disable=invalid-name
		"""Extract pq-grams from trees."""
		return Counter('q_' + pqgram
				for tree in self.unbinarized
					for pqgram in iterpqgrams(tree, p, q))

	def externalfragments(self, tsgfragments, fragmentkeys, bitsets, vocab):
		"""Collect counts for a given set of fragments."""
		corpus = _fragments.getctrees(
				zip(self.trees, self.origsents), vocab=vocab)['trees1']
		results = _fragments.exactcountsslice(
				tsgfragments, corpus, bitsets, indices=0,
				maxnodes=tsgfragments.maxnodes)
		return {'F_%s' % a: c for a, c in zip(fragmentkeys, results)}


def getfeatures(
		path, disc=False, lang='en', start=None, end=None, freqlist=None,
		tsgfragments=None, fragmentkeys=None, bitsets=None, vocab=None):
	"""Collect features for different texts"""
	filenames = OrderedDict((os.path.splitext(os.path.basename(a))[0], a)
			for a in sorted(glob.glob(path)))
	counts = defaultdict(dict)
	for n, (filename, origfilename) in enumerate(list(filenames.items()), 1):
		print('%d. %s' % (n, filename))
		try:
			text = Text.fromparsetrees(
					origfilename, start, end, disc, lang=lang)
		except ValueError as err:
			print('ERROR with %s:\n%s' % (origfilename, err))
			del filenames[filename]
			continue
		# get features that only depend on this text
		# NB: unlike rest, readability scores are not counts!
		# simple features:
		counts['punct'][filename] = text.punctfrequencies()
		counts['pos'][filename] = text.postags()
		counts['read'][filename] = text.readabilityscores(lang)
		counts['basic'][filename] = text.basicfeatures(freqlist, lang)
		# high-dimensional features:
		counts['const'][filename] = text.syntacticcategories()
		counts['unigrams'][filename] = text.wordngrams(1)
		counts['pqgrams'][filename] = text.pqgrams(2, 3)
		counts['prod'][filename] = text.productions(
				parents=False, lexical=True)
		counts['char3grams'][filename] = text.characterngrams(3)
		counts['char4grams'][filename] = text.characterngrams(4)
		counts['bigrams'][filename] = text.wordngrams(2)
		counts['stylebigrams'][filename] = text.stylengrams(2, lang)
		if tsgfragments is not None:
			counts['tsgfrags'][filename] = text.externalfragments(
					tsgfragments, fragmentkeys, bitsets, vocab)
	# put closed class features in a single csv file simple.csv.gz:
	closedclass = ('punct', 'pos', 'read', 'basic')
	featurenames = sorted({(featclass, featlabel)
		for featclass in closedclass
			for filename, featlabels in counts[featclass].items()
				for featlabel in featlabels})
	result = pandas.DataFrame(
			index=pandas.MultiIndex.from_tuples(
				featurenames,
				names=['feat_class', 'feature']),
			columns=filenames)
	for filename in filenames:
		result[filename] = pandas.Series(
				{(featclass, featlabel): val
				for featclass in closedclass
					for featlabel, val in counts[featclass][filename].items()})
	# put other feature types in separate csv files:
	for featclass in counts:
		if featclass not in closedclass:
			with gzip.open('features/%s.csv.gz' % featclass, 'wt') as out:
				prune(pandas.DataFrame(
					counts[featclass], dtype=np.int32).T
					).T.to_csv(out, encoding='utf8')
	return result.T


def fragmentcv(path, target, start=None, end=None, disc=False, numproc=16,
		nonzerocountthreshold=0.5, minfreqthreshold=None,
		relevancythreshold=0.05, redundancythreshold=0.5):
	"""Collect fragment comparisons between folds of the corpus.

	Given folds 1-5, produce files of the form ``FOLDn_FOLDm`` containing the
	fragments in common between fold ``n`` and ``m``, along with the counts of
	those fragments in ``n``.

	:param path: path to files with trees of each text.
	:param target: DataFrame with filenames as index, and columns 'target' with
		numeric target variable to select fragments by,
		and 'fold' which assigns each file to a single fold.
	:param start, end: indices to select slice of trees to use from each file.
	:param numproc: number of processes to use.
	:param nonzerocountthreshold: minimum proportion of non-zero counts for a
		fragment, used as initial filter. None to disable.
	:param minfreqthreshold: minimum overall count (in all texts combined)
		of a feature. None to disable.
	:param relevancythreshold: p-value cutoff for regression test
	:param redundancythreshold: correlation (r) threshold beyond which
		features are considered redundant.
	"""
	filenames = OrderedDict((os.path.splitext(os.path.basename(a))[0], a)
			for a in sorted(glob.glob(path)))
	corpus = treesearch.FragmentSearcher(
			list(filenames.values()), numproc=numproc, inmemory=False)
	# For each fold:
	# 1. collect fragments from fold pairs,
	collectfragments(target, filenames, corpus, start, end, disc, numproc)

	# 2. collect counts in all texts, store in single dataframe
	countfragments(filenames, corpus, start, end)

	# 3. filter fragments and store list for each subset of training folds
	filterfragments(target, filenames, corpus, start, end,
			nonzerocountthreshold, minfreqthreshold, relevancythreshold,
			redundancythreshold)


def collectfragments(target, filenames, corpus, start, end, disc, numproc):
	"""1. collect fragments from text pairs,
	collect counts in all texts of the fold, store in single dataframe.

	Get fragments in common between all pairs of texts in each fold.
	By not comparing sentences within the same text, this avoids extracting
	fragments particular to a single text; however, texts by the same author
	are in the same fold and may lead to author-specific fragments being
	extracted."""
	os.mkdir('features/foldpairs')
	for fold1 in sorted(target.fold.unique()):
		for fold2 in sorted(target.fold.unique()):
			if fold1 >= fold2:  # Compare A-B, but not A-A or B-A.
				continue
			print('fold pair:', fold1, fold2)
			filesinfold1 = [a for a in target[target.fold == fold1].index
					if start is None
					or corpus.getinfo(filenames[a]).len > start]
			filesinfold2 = [a for a in target[target.fold == fold2].index
					if start is None
					or corpus.getinfo(filenames[a]).len > start]
			textpairs = [(filenames[a], filenames[b], start, end, disc)
					for a in filesinfold1
						for b in filesinfold2]
			tmp = set()
			with ProcessPoolExecutor(max_workers=numproc) as pool:
				for n, (pair, frags) in enumerate(
						pool.map(getcommonfragments, textpairs), 1):
					tmp.update(frags)
					print('%d new fragments, %d current, '
							'fold: %s/%s, pair %d/%d: %s'
							% (len(frags), len(tmp),
							fold1, fold2, n, len(textpairs), pair))
			with io.open('features/foldpairs/fold%s_fold%s.txt' % (
					fold1, fold2), 'w', encoding='utf8') as out:
				out.writelines('%s\n' % a for a in sorted(tmp))


def countfragments(filenames, corpus, start, end):
	"""Get counts of fragments across whole corpus; store in sparse matrix."""
	print('merging fold pairs')
	foldfrags = list(uniquestrings(merge(
			*[io.open(a, encoding='utf8')
				for a in glob.glob('features/foldpairs/*.txt')])))
	print('collecting counts for %d fragments' % len(foldfrags))
	fragcounts = scipy.sparse.lil_matrix(
			(len(filenames), len(foldfrags)), dtype=np.int32)
	fileno = {filename: n for n, filename in enumerate(filenames.values())}
	for filename, counts in corpus.batchcounts(foldfrags, start=start, end=end):
		n = fileno[filename]
		fragcounts[n] = np.array(counts, dtype=np.int32)
		print('counts for %d/%d: %s' % (n + 1, len(filenames), filename))
	store_sparse_mat('features/fragcounts.npz',
			scipy.sparse.csc_matrix(fragcounts),
			filenames, foldfrags)


def getcommonfragments(args):
	"""Get list of common fragments given two filenames of files with trees."""
	file1, file2, start, end, disc = args
	# hack: use cached files created by FragmentSearcher object.
	corpus1 = Ctrees.fromfile('%s.ct' % file1)
	corpus2 = Ctrees.fromfile('%s.ct' % file2)
	vocabpath = os.path.join(os.path.dirname(file1), 'treesearchvocab.idx')
	vocab = FixedVocabulary.fromfile(vocabpath)
	frags = _fragments.extractfragments(
			corpus1, start, end, vocab, corpus2, start, end,
			approx=True, disc=disc)
	del corpus1, corpus2
	return (file1, file2), frags


def filterfragments(target, filenames, corpus, start, end,
		nonzerocountthreshold, minfreqthreshold, relevancythreshold,
		redundancythreshold):
	"""2. filter fragments and store list."""
	data, rows, columns = load_sparse_mat('features/fragcounts.npz')
	# FIXME: avoid creating dense matrix
	data = pandas.DataFrame(data.toarray(), index=rows, columns=columns)
	for fold in target.fold.unique():
		# in this iteration, fold is the test fold, and the training set
		# is everything EXCEPT this fold.
		filesnotinfold = [a for a in target[target.fold != fold].index
				if start is None or corpus.getinfo(filenames[a]).len > start]
		print('test fold: %d; %d files' % (fold, len(filesnotinfold)))
		foldfrags = pandas.Index(list(uniquestrings(merge(
				*[io.open(a, encoding='utf8')
					for a in glob.glob('features/foldpairs/*.txt')
					if 'fold%s' % fold not in a]))))
		features = data.loc[filesnotinfold, foldfrags].values
		mask1 = np.ones(len(foldfrags), dtype=np.bool)
		print('all recurring fragments:', len(foldfrags))
		# discard fragments w/zero counts in given % of texts in this fold
		# discard fragments w/frequency below threshold
		if nonzerocountthreshold is not None:
			mask1 &= (features > 0).sum(axis=0) > (
					features.shape[0] * nonzerocountthreshold)
			print('after filter > %d%% texts: %d' % (
					nonzerocountthreshold * 100, mask1.sum()))
		if minfreqthreshold is not None:
			mask1 &= features.sum(axis=0) > minfreqthreshold
			print('after threshold total freq > %d: %d' % (
					minfreqthreshold, mask1.sum()))
		features = features[:, mask1]
		foldfrags = foldfrags[mask1]
		# filter based on correlation w/target values
		_ = selectfragments(
				fold,
				foldfrags,
				features,
				target.ix[filesnotinfold].target,
				relevancythreshold, redundancythreshold)
		del features
		print()


def selectfragments(fold, columns, features, target,
		relevancythreshold=0.05, redundancythreshold=0.5):
	"""Rank features using relevancy measure and greedily select non-redundant
	features.

	:param features: a numpy array <n_samples, n_features> with fragment
		occurrence counts.
	:param target: a vector with the numeric target value of each sample.
	:param relevancythreshold: p-value cutoff for regression test
	:param redundancythreshold: correlation (r) threshold beyond which
		features are considered redundant.
	:returns: tuple ``(indices, pvalues)`` with indices of features that should
		be kept, and the p-values for all features.

	Ranks features using an univariate linear regression or chi2 test to each
	feature wrt. the target value; filters redundant fragments by comparing
	their correlation (continuous) or symmetric uncertainty (discrete)."""
	discrete = isdiscrete(target)
	if discrete:
		# NB: feature values must be non-negative
		# scores, pvalues = feature_selection.chi2(features, target)
		scores, pvalues = feature_selection.f_classif(features, target)
	else:
		scores, pvalues = feature_selection.f_regression(features, target)
	indices = np.argsort(scores)[::-1]
	result = []
	candidates = pvalues < relevancythreshold  # / len(pvalues)
	try:
		counts, bins = np.histogram(pvalues)
	except ValueError:
		pass
	else:
		print('p-value histogram:',
				pandas.Series(counts,
					index=['%4.3f : %4.3f' % a for a in zip(bins, bins[1:])]),
				file=sys.stderr, sep='\n')
	print('after relevancy threshold < %s: %s' % (
			relevancythreshold, features[:, candidates].shape),
			file=sys.stderr)

	# store list of selected fragments, ordered by p-values
	with open('features/rankedfragfold%s.txt' % fold, 'w') as out:
		out.writelines('%s\t%s=%g\n' % (
				columns[n],
				'chi2' if discrete else 'r',
				scores[n] if discrete
				else np.corrcoef(target, features[:, n])[0, 1])
				for n in candidates.nonzero()[0])
	if redundancythreshold is None:
		return candidates.nonzero()[0], pvalues

	# Greedy removal of redundant features
	i = 0  # the i-th best scoring feature
	while i < len(indices):
		n = indices[i]  # the index to the feature
		# TODO: if necessary, replace this linear scan with a proper
		# priority queue
		if not candidates[n]:
			i += 1
			continue
		candidates[n] = False
		result.append(n)
		print('rank: %d, idx: %s, candidates: %s, '
				'%s-score: %s, p-value: %s, fragment:\n%s' % (
				i, n, len(candidates.nonzero()[0]),
				'chi2' if discrete else 'F',
				scores[n], pvalues[n],  # features[:, n].corr(target),
				columns[n]), file=sys.stderr)
		x = 0
		for m in candidates.nonzero()[0]:
			if discrete:
				sim = symmetricuncertainty(features[:, n], features[:, m])
			else:
				sim = abs(np.corrcoef(features[:, n], features[:, m])[0, 1])
			if sim > redundancythreshold:
				candidates[m] = False
			if i < 100 and x < 5:
				print('red: %s (%s)\n%s\n' % (
						sim > redundancythreshold, sim, columns[m]),
						file=sys.stderr)
			x += 1
		i += 1
	print('after redundancy removal > %s: %s' % (
			redundancythreshold, len(result)), file=sys.stderr)

	# store list of selected fragments, ordered by p-values
	with open('features/rankednonredundantfragfold%s.txt' % fold, 'w') as out:
		out.writelines('%s\t%s=%g\n' % (
				columns[n],
				'chi2' if discrete else 'r',
				scores[n] if discrete
				else np.corrcoef(target, features[:, n])[0, 1])
				for n in result)
	return result, pvalues


def prune(df):
	"""Remove useless features (columns) from dataframe."""
	# drop features that occur in only one text:
	df = df.loc[:, (df > 0).sum(axis=0) > 1]
	# drop features that do not occur more than once in any text
	df = df.loc[:, df.max(axis=0) > 1]
	# replace missing data with zero counts
	df.fillna(0, inplace=True)
	return df


def addparents(tree):
	"""Return a copy of tree with parent annotation."""
	tree = ParentedTree.convert(tree)
	for node in tree.postorder():
		if node is tree:
			continue
		node.label += '^' + node.parent.label
	return tree


def cfgproductions(tree, sent, parents=False, lexical=False):
	"""Yield productions of tree as strings."""
	for node in tree.subtrees():
		if not node:
			continue
		elif isinstance(node[0], int):
			if not lexical:
				continue
			rhs = sent[node[0]]
		else:
			rhs = ' '.join(a.label for a in node)
		lhs = (('%s^%s' % (node.label, node.parent.label))
				if parents and node is not tree
				else node.label)
		yield '%s => %s' % (lhs, rhs)


def iterpqgrams(tree, p, q):  # pylint: disable=invalid-name
	"""Traverse tree and yield pq-grams at each node.

	Terminals are not included. ``tree`` should be a ParentedTree."""
	for node in tree.subtrees():
		ancestors = []
		x = node
		for _ in range(p):
			if x.parent is None:
				ancestors.extend(['*'] * (p - len(ancestors)))
				break
			ancestors.append(x.parent.label)
			x = x.parent
		ancestors = ','.join(ancestors)
		siblings = [x.label for x in node if isinstance(x, Tree)]
		if not siblings:
			continue
		for siblingngram in ngrams(siblings, q,
				pad_left=True, pad_right=True,
				left_pad_symbol='*', right_pad_symbol='*'):
			yield '|<%s>^<%s>' % (','.join(siblingngram), ancestors)


def detokenize(sent):
	"""Transform string of space-separated tokens into sentence with regular
	spacing.

	Based on detokenizer.perl from Moses SMT.

	>>> detokenize("' I don 't know ' , he said .")
	"'I don't know', he said."
	"""
	sent = " " + sent + " "

	# merge de-contracted forms except where the second word begins with an
	# apostrophe (those are handled later)
	sent = re.sub(" n't ", "n't ", sent)
	sent = re.sub(" N'T ", "N'T ", sent)
	sent = re.sub(" ([Cc])an not r", r" \1annot ", sent)
	sent = re.sub(" ([Dd])' ye ", r" \1'ye ", sent)
	sent = re.sub(" ([Gg])im me ", r" \1imme ", sent)
	sent = re.sub(" ([Gg])on na ", r" \1onna ", sent)
	sent = re.sub(" ([Gg])ot ta ", r" \1otta ", sent)
	sent = re.sub(" ([Ll])em me ", r" \1emme ", sent)
	sent = re.sub(" '([Tt]) is ", r" '\1is ", sent)
	sent = re.sub(" '([Tt]) was ", r" '\1was ", sent)
	sent = re.sub(" ([Ww])an na ", r" \1anna ", sent)

	# restore brackets
	sent = re.sub('-LRB-', '(', sent)
	sent = re.sub('-RRB-', ')', sent)
	sent = re.sub('-LSB-', '[', sent)
	sent = re.sub('-RSB-', ']', sent)
	sent = re.sub('-LCB-', '{', sent)
	sent = re.sub('-RCB-', '}', sent)

	words = sent.split()
	result = []
	prependspace = ' '
	inquote = False
	for i, word in enumerate(words):
		if re.match('^[$£€([{¿¡]+$', word, re.UNICODE):
			result.append(prependspace)
			result.append(word)
			prependspace = ''
		elif re.match(r'^[,.?!:;\\%})\]]+$', word, re.UNICODE):
			result.append(word)
			prependspace = ' '
		# NB: first \w should not include digits
		elif (i and re.match(r"^'\w", word, re.UNICODE)
				and re.match(r'.*\w$', words[i - 1], re.UNICODE)):
			result.append(word)
			prependspace = ' '
		elif not inquote and word in '"``\'\'':
			inquote = True
			result.append(prependspace)
			result.append(word)
			prependspace = ''
		elif inquote and word in '"\'\'':
			inquote = False
			result.append(word)
			prependspace = ' '
		else:
			result.append(prependspace)
			result.append(word)
			prependspace = ' '
	return ''.join(result).strip()


def isdiscrete(vec):
	"""Heuristically test if vector is discrete or continuous."""
	return len(vec.unique()) < 0.25 * len(vec)


def symmetricuncertainty(vec1, vec2, epsilon=0.01):
	"""Compute symmetric uncertainty between two discrete vectors.

	:param vec1, vec2: numpy arrays with probabilities or counts; counts are
		automatically normalized and zeroes are replaced with epsilon.
	:returns: a float in the range [0, 1]; higher values indicate a higher
		similarity of vec1 and vec2."""
	vec1 = vec1 + epsilon
	vec2 = vec2 + epsilon
	infgain = entropy(vec1, vec2)
	return 2 * infgain / (entropy(vec1) + entropy(vec2))


def sparse_corrcoef(A):
	"""Compute a correlation matrix for a sparse matrix."""
	A = A.astype(np.float64)
	# compute the covariance matrix
	# (see http://stackoverflow.com/questions/16062804/)
	A = A - A.mean(1)
	norm = A.shape[1] - 1.
	C = A.dot(A.T.conjugate()) / norm

	# the correlation coefficients are given by C_{i,j} / sqrt(C_{ii} * C_{jj})
	d = np.diag(C)
	coeffs = C / np.sqrt(np.outer(d, d))
	return coeffs


def readexternalfragments(filename, disc=False):
	"""Read a set of predefined tree fragments.	"""
	items, fragmentkeys, vocab = [], [], Vocabulary()
	for a, b, _ in treebank.incrementaltreereader(io.open(filename)):
		fragmentkeys.append(a)
		items.append((binarize(handledisc(a), dot=True), b))
	tsgfragments = _fragments.getctrees(items, vocab=vocab)['trees1']
	_fragmentkeys, bitsets = _fragments.completebitsets(
			tsgfragments, vocab, tsgfragments.maxnodes, disc=disc)
	return dict(tsgfragments=tsgfragments, fragmentkeys=fragmentkeys,
			bitsets=bitsets, vocab=vocab)


def uniquestrings(iterable):
	"""Yield unique strings from a sorted iterable.

	Also strip trailing newlines/whitespace."""
	prev = None
	for line in iterable:
		frag = line.rstrip()
		if frag != prev:
			if prev is not None:
				yield prev
			prev = frag
	if prev is not None:
		yield prev


def store_sparse_mat(filename, data, rows, columns):
	"""Store scipy.sparse csc matrix in .npz format,
	along with corresponding row/column labels."""
	assert data.__class__ == scipy.sparse.csc.csc_matrix, (
			'This code only works for csc matrices')
	np.savez_compressed(filename,
			data=data.data,
			indices=data.indices,
			indptr=data.indptr,
			shape=data.shape,
			rows='\n'.join(rows).encode('utf8'),
			columns='\n'.join(columns).encode('utf8'))


def load_sparse_mat(filename):
	"""Load scipy.sparse csc matrix stored as numpy arrays."""
	inp = np.load(filename)
	pars = (inp['data'], inp['indices'], inp['indptr'])
	data = scipy.sparse.csc_matrix(pars, shape=inp['shape'], dtype=np.int32)
	rows = pandas.Index(bytes(inp['rows']).decode('utf8').splitlines())
	columns = pandas.Index(bytes(inp['columns']).decode('utf8').splitlines())
	return data, rows, columns


class OrderedLabelKFold(cross_validation._BaseKFold):
	"""K-fold iterator variant with non-overlapping labels.

	The same label will not appear in two different folds (the number of
	distinct labels has to be at least equal to the number of folds).

	Each k-th label ends up in the k-th fold, where k is the index of where the
	label is first encountered in ``labels``.

	This code is based on LabelKFold from scikit-learn.

	Parameters
	----------
	labels : array-like with shape (n_samples, )
		Contains a label for each sample.
		The folds are built so that the same label does not appear in two
		different folds.

	n_folds : int, default=3
		Number of folds. Must be at least 2.

	shuffle : boolean, optional
		Whether to shuffle the data before splitting into batches.

	random_state : None, int or RandomState
		When shuffle=True, pseudo-random number generator state used for
		shuffling. If None, use default numpy RNG for shuffling.

	Examples
	--------
	>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
	>>> y = np.array([1, 2, 3, 4])
	>>> labels = np.array([0, 0, 2, 2])
	>>> label_kfold = OrderedLabelKFold(labels, n_folds=2)
	>>> len(label_kfold)
	2
	>>> print(label_kfold)
	OrderedLabelKFold(n_labels=4, n_folds=2)
	>>> for train_index, test_index in label_kfold:
	...     print("TRAIN:", train_index, "TEST:", test_index)
	...     X_train, X_test = X[train_index], X[test_index]
	...     y_train, y_test = y[train_index], y[test_index]
	...     print(X_train, X_test, y_train, y_test)
	...
	TRAIN: [0 1] TEST: [2 3]
	[[1 2]
	 [3 4]] [[5 6]
	 [7 8]] [1 2] [3 4]
	TRAIN: [2 3] TEST: [0 1]
	[[5 6]
	 [7 8]] [[1 2]
	 [3 4]] [3 4] [1 2]

	See also
	--------
	LeaveOneLabelOut for splitting the data according to explicit,
	domain-specific stratification of the dataset.
	"""
	def __init__(self, labels, n_folds=3, shuffle=False, random_state=None):
		super(OrderedLabelKFold, self).__init__(len(labels), n_folds,
				shuffle, random_state)

		unique_labels, unique_indices, unique_inverse = np.unique(
				labels, return_index=True, return_inverse=True)
		n_labels = len(unique_labels)

		if n_folds > n_labels:
			raise ValueError(
					("Cannot have number of folds n_folds={0} greater"
					" than the number of labels: {1}.").format(
					n_folds, n_labels))

		# indices of labels in order of first occurrence
		ranking = np.argsort(unique_indices)

		if shuffle:
			# When shuffle=True, label names are randomized to obtain random
			# fold assigments.
			rng = check_random_state(self.random_state)
			rng.shuffle(ranking)

		# Weight labels by their number of occurences
		n_samples_per_label = np.bincount(unique_inverse)

		# Total weight of each fold
		n_samples_per_fold = np.zeros(n_folds, dtype=np.intp)

		# Mapping from label index to fold index
		label_to_fold = np.zeros(n_labels, dtype=np.intp)

		for n in range(n_labels):
			# Assign this label to the fold that currently has the least
			# number of samples
			fold = np.argmin(n_samples_per_fold)
			n_samples_per_fold[fold] += n_samples_per_label[ranking[n]]
			label_to_fold[ranking[n]] = fold

		self.idxs = label_to_fold[unique_inverse]

	def _iter_test_indices(self):
		for f in range(self.n_folds):
			yield (self.idxs == f).nonzero()[0]

	def __repr__(self):
		return '{0}.{1}(n_labels={2}, n_folds={3})'.format(
			self.__class__.__module__,
			self.__class__.__name__,
			self.n,
			self.n_folds,
		)

	def __len__(self):
		return self.n_folds


def main():
	"""Extract features from a corpus."""
	options = ('lang=', 'slice=', 'extfrags=', 'freqlist=', 'numproc=',
			'folds=', 'nonzerocount=', 'minfreq=', 'relevancy=', 'redundancy=',
			'disc', 'nofragments')
	try:
		opts, args = gnu_getopt(sys.argv[1:], '', options)
		datasetdir, targetcolumn = args
	except (GetoptError, ValueError) as err:
		print('error:', err, file=sys.stderr)
		print(__doc__)
		sys.exit(2)
	opts = dict(opts)
	os.chdir(datasetdir)

	# Sort by target variable to get stratified folds.
	metadata = pandas.read_csv('metadata.csv').sort_values(by=targetcolumn)
	if 'fold' not in metadata.columns:
		numfolds = 5

		# Assign each book to a fold, but make sure that books by same author
		# end up in same fold.
		# NB: assumes labels are of the form `AuthorLastName_Title`!
		authors = metadata.Label.str.split('_').str[0]
		cv = OrderedLabelKFold(
				authors, n_folds=numfolds, shuffle=False)  # , random_state=1)
		metadata['fold'] = cv.idxs.astype(int) + 1  # start fold IDs at 1

	os.mkdir('features')
	target = pandas.DataFrame(
			data={'fold': metadata.fold, 'target': metadata[targetcolumn]},
			index=metadata.index)
	target.index = metadata.Label
	target.to_csv('features/target.csv', encoding='utf8')

	start, end = opts.get('--slice', ':').split(':')
	start, end = (int(start) if start else 0), (int(end) if end else 1000)
	filepattern = ('parses/*.mrg' if glob.glob('parses/*.mrg')
			else 'parses/*.dbr')
	if '--extfrags' in opts:
		extfrags = readexternalfragments(
				opts.get('--extfrags'),
				disc='--disc' in opts)
	else:
		extfrags = {}
	if '--freqlist' in opts:
		freqlist = pandas.read_table(
				opts['--freqlist'], encoding='utf8', index_col=0, header=None,
				names=['word', 'count'])
	else:
		freqlist = None

	data = getfeatures(
			filepattern,
			disc='--disc' in opts,
			lang=opts.get('--lang', 'en'),
			start=start, end=end,
			freqlist=freqlist,
			**extfrags)
	with gzip.open('features/simple.csv.gz', 'wt') as out:
		data.to_csv(out, encoding='utf8')

	if '--nofragments' in opts:
		return
	if opts.get('--nonzerocount') == 'None':
		nonzerocountthreshold = None
	else:
		nonzerocountthreshold = float(opts.get('--nonzerocount', 0.05))
	if opts.get('--minfreq') == 'None':
		minfreqthreshold = None
	else:
		minfreqthreshold = int(opts.get('--minfreq', 50))

	fragmentcv(
			filepattern, target, start, end,
			disc='--disc' in opts,
			numproc=int(opts.get('--numproc', 1)),
			nonzerocountthreshold=nonzerocountthreshold,
			minfreqthreshold=minfreqthreshold,
			relevancythreshold=float(opts.get('--relevancy', 0.05)),
			redundancythreshold=float(opts.get('--redundancy', 0.5)))


if __name__ == '__main__':
	faulthandler.enable()
	main()
