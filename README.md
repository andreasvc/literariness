Code for the paper "A data-oriented model of literary language"
===============================================================

This is the code as used in the EACL 2017 paper to predict whether a text would
be rated as literary or not; however, the code is general enough to predict any
continuous or discrete variable from various predefined textual features, including
syntactic tree fragments.

Expects a corpus of parsed texts and extracts features for a regression or
classification task on them, as well as running a cross-validated evaluation
of models trained on those features.

Requirements
------------

Python 3.3+

    $ pip3 install --user pandas scikit-learn nltk readability disco-dop

Feature extraction
------------------

    $ python3 features.py <dataset> <options>

Run without arguments for a description of available options and expected input.

Predictive model
----------------

    $ python3 predict.py <dataset>

Run without arguments for a description of expected input.

Dataset used in paper
---------------------
For obvious copyright reasons, the dataset of recent novels unfortunately
cannot be made available. The following is the output of the experiments as
reported in the paper:

    $ python3 features.py Riddle "Literary rating" --disc --lang=nl --freqlist=sonar-word.freqsort.lower.gz --slice=1000:2000
    [...]
    $ python3 predict.py Riddle

    bigrams
    (369, 72209) (369, 11625)
                      mean  std err
    $R^2$           55.800    3.000
    Kendall $\tau$   0.525    0.025
    RMS error        0.666    0.191

    char3grams
    (369, 17239)
                      mean  std err
    $R^2$           51.200    3.200
    Kendall $\tau$   0.493    0.020
    RMS error        0.699    0.184

    char4grams
    (369, 65562)
                      mean  std err
    $R^2$           56.900    2.400
    Kendall $\tau$   0.522    0.016
    RMS error        0.657    0.162

    fragments
    (369, 4287512)
    1 (297, 7779) (72, 7779)
    2 (293, 8130) (76, 8130)
    3 (296, 7602) (73, 7602)
    4 (293, 7277) (76, 7277)
    5 (297, 8108) (72, 8108)
                      mean  std err
    $R^2$           57.200    3.300
    Kendall $\tau$   0.518    0.025
    RMS error        0.655    0.191

                      1       2       3       4       5
    bigrams     59.800  47.003  58.045  63.606  50.743
    char4grams  58.571  50.440  54.153  65.014  56.168
    fragments   61.501  53.382  58.690  65.752  46.453

                                 $R^2$  Kendall $\tau$  RMS error
    words per sent.               16.4           0.238      0.916
    + % direct speech sentences   23.1           0.293      0.879
    + top3000vocab                23.5           0.291      0.876
    + bzip2_ratio                 24.4           0.298      0.871
    + cliches                     30.0           0.349      0.838
    + topics                      52.2           0.614      0.692
    + bigrams                     59.5           0.640      0.640
    + char4grams                  59.9           0.638      0.638
    + fragments                   61.2           0.638      0.626
    + Category                    74.3           0.667      0.509
    + Translated                  74.0           0.666      0.512
    + Gender                      76.0           0.670      0.492


Example Gutenberg dataset
-------------------------
A set of 100 novels from project Gutenberg. The prediction target is the download count.
This is a subset of the dataset used in Ashok et al. (2013, EMNLP), with similar features.
This only serves as a toy corpus for demonstration. The low scores are probably
due to the small dataset and the prediction target being too noisy (evaluating as
binary classification would be more appropriate).

    $ curl -sSL https://staff.fnwi.uva.nl/a.w.vancranenburgh/100gutenbergnovels.tar.bz2 | tar -xjf -
    [...]
    $ python3 features.py Gutenberg DownloadCount --numproc=16 --lang=en --minfreq=10
    [...]
    $ python3 predict.py Gutenberg
    [...]
    N = 100
                             $R^2$  Kendall $\tau$  RMS error
    bigrams                  8.198           0.311    201.304
    + fragments              9.182           0.316    199.648
    + stylebigrams           8.618           0.296    199.912
    + pqgrams               15.404           0.352    194.062
    + prod                  15.377           0.331    194.380
    + const                 15.030           0.337    194.607
    + pos                    7.466           0.331    201.308
    + punct                 -1.415           0.255    209.573
    + read                  -0.511           0.272    208.246
    + r_words_per_sentence  -0.511           0.272    208.246
    + b_%_direct_speech     -0.491           0.272    208.137
    + b_bzip2_ratio         -1.332           0.261    209.069
    + b_avgdeplen           -1.111           0.261    208.886


Inspecting induced fragments:

    $ discodop treedraw < Gutenberg/features/rankednonredundantfragfold1.txt | less -R
    1. (len=6):     r=0.590169
                 SBAR
             ┌────┴─────────────────────┐
            SBAR                        │
     ┌───────┴────┐                     │
     │            S                     │
     │   ┌────────┴────┐                │
     │   │             VP         SBAR|<SBAR.CC,
     │   │             │              SBAR>
     │   │        ┌────┴───┐   ┌────────┴─────────┐
     IN  NP      AUX       VP  CC                SBAR
     │   │        │        │   │                  │
    ... ...      ...      ... and                ...

    2. (len=5):     r=0.523414
             VP
     ┌───────┴───┐
     │           VP
     │   ┌───────┴───┐
     │   │           VP
     │   │   ┌───────┴────┐
     │   │   │            PP
     │   │   │       ┌────┴───┐
     MD AUX VBN      IN       NP
     │   │   │       │        │
    ...  be ...     from     ...

    3. (len=7):     r=0.522175
          PP
     ┌────┴───────┐
     │            NP
     │    ┌───────┴───────┐
     │    │               PP
     │    │   ┌───────────┴───┐
     │    │   │               NP
     │    │   │       ┌───────┴───────┐
     │    │   │       NP              PP
     │    │   │   ┌───┴───┐       ┌───┴───┐
     IN   NP  IN  DT      NN      IN      NP
     │    │   │   │       │       │       │
    with ... ... ...     ...      of     ...


Reference
---------

    @inproceedings{vancranenburgh2017literary,
        title={A data-oriented model of literary language},
        author={van Cranenburgh, Andreas and Rens Bod},
        booktitle={Proceedings of EACL},
        year={2017},
        pages={...},
        url={...},
    }

