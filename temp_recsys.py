"""
import nltk
nltk.download(['punkt', 
               'averaged_perceptron_tagger', 
               'stopwords',
               'wordnet',
               'omw-1.4',
               ],
              quiet=True, 
              raise_on_error=True,
              )
print(nltk.corpus.stopwords.fileids())

#STOPWORDS = set(nltk.corpus.stopwords.words('english'))
STOPWORDS = set(nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids()))

print(len(STOPWORDS), STOPWORDS)

MIN_WORDS = 4
MAX_WORDS = 200

PATTERN_S = re.compile("\'s")  # matches `'s` from text  
PATTERN_RN = re.compile("\\r\\n") #matches `\r` and `\n`
PATTERN_PUNC = re.compile(r"[^\w\s]") # matches all non 0-9 A-z whitespace 
def clean_text(text):
    """
    Series of cleaning. String to lower case, remove non words characters and numbers.
        text (str): input text
    return (str): modified initial text
    """
    text = text.lower()  # lowercase text
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    text = re.sub(PATTERN_PUNC, ' ', text)
    return text

def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
    """
    Lemmatize, tokenize, crop and remove stop words.
    """
    if lemmatize:
        stemmer = nltk.stem.WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in nltk.tokenize.word_tokenize(sentence)]
    else:
        tokens = [w for w in nltk.tokenize.word_tokenize(sentence)]
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                                        and w not in stopwords)]
    return tokens    

def clean_sentences(df):
    """
    Remove irrelavant characters (in new column clean_sentence).
    Lemmatize, tokenize words into list of words (in new column tok_lem_sentence).
    """
    print('Cleaning sentences...')
    df['clean_sentence'] = df['sentence'].apply(clean_text)
    df['tok_lem_sentence'] = df['clean_sentence'].apply(
        lambda x: tokenizer(x, 
                            min_words=MIN_WORDS, 
                            max_words=MAX_WORDS, 
                            stopwords=STOPWORDS, 
                            lemmatize=True,
                            )
        )
    return df

def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]  
    return best_index

def get_recommendations_tfidf(phrase, topN=5):
    """
    Return the database phrases in order of highest cosine similarity relatively to each 
    token of the target phrase. 
    """
    print(f"{'RecSys (TFIDF)'.center(80, '-')}")
    # Adapt stop words
    token_stop = tokenizer(' '.join(STOPWORDS), lemmatize=True) # orig: False
    print(f"tokenizer stop words: ({len(token_stop)})\n{token_stop}")

    # Fit TFIDF
    tfidf = TfidfVectorizer(#min_df=5,
                                #ngram_range=(1, 2),
                                #tokenizer=Tokenizer(),
                                tokenizer=tokenizer,
                                stop_words=token_stop,
                                )
    #tfidf_mat = tfidf.fit_transform(df['phrase'].values)
    tfidf_mat = tfidf.fit_transform(df['clean_phrase'].values)

    print(tfidf.get_feature_names_out()[:30])
    print(tfidf_mat.shape)  # (n_sample, n_vocab))
    #print(json.dumps(tfidf.vocabulary_, indent=2, ensure_ascii=False))

    # Embed the query phrase
    tokens = [str(tok) for tok in tokenizer(phrase)]
    print(f">> tokenize >> {phrase} <<\t{len(tokens)} {tokens}")

    vec = tfidf.transform(tokens)
    print(vec.shape)
    #print(vec.toarray())

    # Create list with similarity between query and dataset
    mat = cosine_similarity(vec, tfidf_mat)
    # Best cosine distance for each token independantly
    print(mat.shape)

    best_index = extract_best_indices(mat, topk=topN)

    return best_index

"""
