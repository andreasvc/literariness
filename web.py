"""Web interface for applying predictive models to given sample of text."""
# stdlib
from __future__ import print_function, absolute_import
import os
import sys
import glob
import logging
import subprocess
from functools import wraps
from collections import OrderedDict
# import matplotlib
# matplotlib.use('AGG')
# import matplotlib.cm as cm
import pandas
import re2  # https://www.github.com/andreasvc/pyre2
# Flask & co
from flask import Flask, Response
from flask import request, render_template
import joblib
# local imports
from features import Text

DEBUG = True  # when True: enable debugging interface, disable multiprocessing
APP = Flask(__name__)
STANDALONE = __name__ == '__main__'


@APP.route('/')
@APP.route('/index')
def index():
	"""Start page where a text can be entered."""
	return render_template('index.html')


@APP.route('/results', methods=('POST', ))
def results():
	"""Results page for given text."""
	if 'text' not in request.form:
		return 'No form'
	text = request.form['text']
	tokens = tokenize(text)
	feat = extractfeatures(tokens)
	feat['Category'] = request.form['Category']
	feat['Translated'] = request.form['Translated']
	feat['Author gender'] = request.form['Gender']
	pred = getpredictions(feat)
	return render_template('predictions.html',
			tokens=tokens, feat=feat, pred=pred)


def tokenize(text):
	# FIXME: more extensive cleaning
	text = text.replace('\n', ' ')
	proc = subprocess.Popen(
			[os.path.join(
				os.getenv('ALPINO_HOME'), 'Tokenization/tokenize.sh')],
			stdin=subprocess.PIPE, stdout=subprocess.PIPE,
			stderr=subprocess.PIPE)
	out, err = proc.communicate(text.encode('utf8'))
	return out.decode('utf8')


def extractfeatures(tokens):
	result = OrderedDict()
	text = Text.fromtokenized(
			'input', tokens, start=0, end=1000, maxlen=300, lang='nl')
	read = text.readabilityscores()
	basic = text.basicfeatures(freqlist=FREQLIST)
	result['sentences'] = read['r_sentences']
	result['words per sentence'] = read['r_words_per_sentence']
	result['% direct speech sentences'] = basic['b_%_direct_speech']
	result['top3000vocab'] = basic['b_top3000vocab']
	result['bzip2_ratio'] = basic['b_bzip2_ratio']
	matches = sum(query.count(tokens) for query in CLICHES)
	result['cliches'] = 10000.0 * matches / max(tokens.count('\n'), 1)
	result['bigrams'] = text.wordngrams(2)
	result['char4grams'] = text.characterngrams(4)
	return result


def getpredictions(feat):
	result = OrderedDict()
	for featclass in ('bigrams', 'char4grams'):
		vec = pandas.Series(
				feat[featclass], index=FEATURENAMES[featclass]).fillna(0)
		result[featclass] = MODELS[featclass].predict([vec.values])[0]
	feat['bigrams'] = result['bigrams']
	feat['char4grams'] = result['char4grams']
	result['ensemble'] = MODELS['ensemble'].predict(feat)[0]
	return result


class QueryStringRedirectMiddleware(object):
	"""Support ; as query delimiter.

	http://flask.pocoo.org/snippets/43/"""
	def __init__(self, application):
		self.application = application

	def __call__(self, environ, start_response):
		qs = environ.get('QUERY_STRING', '')
		environ['QUERY_STRING'] = qs.replace(';', '&')
		return self.application(environ, start_response)


APP.wsgi_app = QueryStringRedirectMiddleware(APP.wsgi_app)

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.handlers[0].setFormatter(logging.Formatter(
		fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
log.info('loading.')
if STANDALONE:
	from getopt import gnu_getopt, GetoptError
	try:
		opts, _args = gnu_getopt(sys.argv[1:], '',
				['port=', 'ip=', 'numproc=', 'debug'])
		opts = dict(opts)
	except GetoptError as err:
		print('error: %r' % err, file=sys.stderr)
		sys.exit(2)
	DEBUG = '--debug' in opts
FREQLIST = pandas.read_table('sonar-word.freqsort.lower.gz',
		encoding='utf8', index_col=0, header=None, names=['word', 'count'],
		nrows=20000)  # remove this to get accurate 'unknownvocab' results
re2.set_fallback_notification(re2.FALLBACK_EXCEPTION)
with open('cliche_queries.txt', encoding='utf8') as inp:
	queries = inp.read().splitlines()
# precompile into chunks of 500 cliche expressions
CLICHES = [re2.compile(
			'|'.join(query for query in queries[n:n + 500]),
			flags=re2.UNICODE | re2.IGNORECASE | re2.MULTILINE,
			max_mem=8 << 26)  # 500 MB
		for n in range(0, len(queries), 500)]
# pre-load data/models
FEATURENAMES = {}
MODELS = dict(ensemble=joblib.load('Riddle/ensemble.pkl'))
FEATURENAMES['bigrams'], MODELS['bigrams'] = joblib.load(
		'Riddle/features/bigrams.pkl')
FEATURENAMES['char4grams'], MODELS['char4grams'] = joblib.load(
		'Riddle/features/char4grams.pkl')
log.info('done.')
if STANDALONE:
	APP.run(use_reloader=True,
			host=opts.get('--ip', '0.0.0.0'),
			port=int(opts.get('--port', 5004)),
			debug=DEBUG)
