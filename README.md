
# Text similarity with Python. A simple example for a baseline similarity measure between short texts for social science analysis.

In the following notebook, I cover the most basic approaches to pre-processing short texts and vectorizing them in a bag-of-words fashion. This should allow, as we see at the end, to compute similarity metrics between textual data potentially enriching the output of any social-science analysis. 

This notebook is by no means a complete solution, nor intends to be one, and should just be taken as a minimal "how-to". The specificities of each data source, lenght of text and depth of analysis should be taken into account when pre-processing text.

Extensive documentation is available from the developpers of each library on each method. Therfore, I don't discuss hyperparameters (when required) beyond the application or what each function does or how. 

## Pre-processing text.
This block imports the main libraries that we're going to use for NLP. Gensim is a very complete toolkit (that actually would allow to do all of the rest, but using NLTK for convenience, since it is widely used in the community.
```python

import gensim
from nltk.corpus import stopwords
import numpy
import pickle

from nltk.stem import PorterStemmer 
```


The following line imports to a list the stopwords in English. Being a standard package, this includes the most common cases, 
```python
StopWords = set(stopwords.words('english'))
```
but we can customize this list, append new characters, or create an entirely new list which we also filter words from:
```python
customStopWords = ["omar","is","handsome"]
```   
This function Tokenizes Raw Text Input (i.e. a paragraph without any prior pre-processing). 
The first line of code does (in this order):
- Removes dots and replaces them with blankspace.
- Splits the text by blankspaces
- Removes blankspaces adjacent to words
- Sets words to lowercase
The second line inside de function transforms the text into a list omitting the previously set stopwords.
```python
def TokenizeText(RawText):
    PreProcessedText = [word.strip().lower() for word in RawText.replace('.',' ').split()]
    TokenizedText = [word for word in PreProcessedText if word not in StopWords]
    return TokenizedText
```
We can extend the previous function with additional lines in order to remove, for example, **custom stopwords, special characters, short words** depending on the specificities of our data.
```python
    #Adding the line below would additionally remove the list of Custom stopwords
    TokenizedText = [word for word in PreProcessedText if word not in customStopWords]

    #The line below would remove words comprised of one character
    TokenizedText = [word for word in TokenizedText if len(word)>1]
    
    #The following would remove any end-of-word characters in the given list (instead of compressed list, written as a for loop for clarity of code:
    TokenizedText2=[]
    for token in TokenizedText:
        if token[-1] in [":",",","?",".",";","!",")","'",'"',"]"]:
            token=token[:-1]
        TokenizedText2.append(token)
    TokenizedText = [word for word in TokenizedText2]
```

#### Usage example. 
```python
txt1="The following text is an example of an abstract. It should allow us to see how the functions work, and especially, develop the next steps in text processing for vectorization."
txt2="Normally text will come from a database, which we organize around an iterable, in this case, a list of texts. This can also be streamed directly from a DB"
data=[txt1,txt2]
```
Two "abstracts" have been loaded into an iterable (a list, in this example). We apply the function "TokenizeText" onto the first, to see the output:
```python
TokenizeText(txt1)
>>> ['following',
 'text',
 'example',
 'abstract',
 'allow',
 'us',
 'see',
 'functions',
 'work',
 'especially',
 'develop',
 'next',
 'steps',
 'text',
 'processing',
 'vectorization']
```
We then need to apply the function sequentially to the iterable. This text is already ready for **TF** or **TF-IDF**, construction. For that we can use either Gensim or Sci-kit Learn libraries.

## More pre-processing: Building n-grams of the most common co-occurence of words.
There are several approaches that accomplish a similar task. If we have a dictionary of phrases or technical words that belong together, we can pre-load them into NLTK. This step has to be introduced in the TokenizeText function defined above. In the next lines, to se how it works, we run it on txt1. The output will be a list of words where "next" and "steps" are one token instead of two.
```python
from nltk.tokenize import MWETokenizer
tokenizer = MWETokenizer()
tokenizer.add_mwe(("next","steps"))
tokenizer.tokenize(txt1.split())
>>> ['The',
 'following',
 'text',
 'is',
 'an',
 'example',
     ...
 'next_steps',
 'in',
 'text',
 'processing',
 'for',
 'vectorization.']
```
This isn't always the case, and if we don't know the specificities of our data (or it is very techical) it's best to use different methods learnt from the actual data. It is very useful to benchmark whether $P("Next Steps") > P("Next")*P("Steps")$. There are many pre-trained models about this, but good ones are sometimes complex to use and only capable on "standard" text sources. Using simple information metric measures and Gensim, we can perform a similar trick. The example provided below learns normalized pointwise mutual information on the co-ocurrence of words, and joins those that are above the threshold. It takes as an input already-tokenized text.
```python
phrases=gensim.models.phrases.Phrases([TokenizeText(txt) for txt in data],min_count=1,scoring="npmi",threshold=0.2)
bigram=gensim.models.phrases.Phraser(phrases)
```
We can then apply the learnt "bigram" over any list of tokenized text, and the output will include as single tokens the united pairs of words. For higher-order ngrams, we need to iteratively apply the previous method. (In this example, most words will become part of pairs, given that the sample of texts is very limited).
```python
bigram[TokenizeText(txt1)]

>>> ['following_text',
 'example_abstract',
 'allow_us',
 'see_functions',
 'work_especially',
 'develop_next',
 'steps_text',
 'processing_vectorization']
```

## Term-frequency (TF) and TF-iDF (-inverse Document Frequency) and text similarity.
First, we need to import a few other libraries in order to make things easier.
```python
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
I'm going to add another "fake" abstract to the ones we had, which shares some words with the first, and I'm going to incorporate it into our iterable. This is just to exemplify with a result at the end.
```python
txt3="The following text is another example of an abstract very similar to the first."
data.append(txt3)
```
The following lines of code do (in order):
- A count of the tokens present in each text from the iterable after tokenization, and multi-word (bigram)-ization.
- A transformation into a sparse vector readable for Sci-kit Learn
- Storage into variable "names" of the meaning of each column in the vector space.

- tf-idf and tf trainig on the previous (Last two blocks, one thing each block)

```python
WordCounts=[Counter(bigram[TokenizeText(txt)]) for txt in data]
v= DictVectorizer()
ved = v.fit_transform(WordCounts)
names = v.get_feature_names()

tfidf_transformer = TfidfTransformer(use_idf=True).fit(ved)
tfidf = tfidf_transformer.transform(ved)

tf_tranformer=TfidfTransformer(use_idf=False,norm=None).fit(ved)
tf=tf_tranformer.fit_transform(ved)
```

The output for either of these Objects (tf / tfidf) is a sparse matrix where each column is a token, and each row is a document. In our case, this should be a matrix of shape 3 (three documents in our data iterable) and **n** tokens
```python
print(tfidf.shape)
>>>(3, 18)
```
Hence, we can directly apply a metric function in order to measure the similarity amongst these vectors. This results in a diagonal matrix where all the document-vectors are compared to all. The diagonal should normally be 1 (totally similar to itself), and since we introduced a third text that shares words with the first, we should see higher similarity between the first and the third than any other pair:
```python
cosine_similarity(tf)
>>>array([[1.       , 0.       , 0.1767767],
       [0.       , 1.       , 0.       ],
       [0.1767767, 0.       , 1.       ]])
```


** Several further characteristics might be incorporated when building tf/tf-idf with TfidfTransformer:
- min_df: minimum number of word occurence across all documents
- max_df: max number of word occurence across all documents
- max_features: only consider top max_features ordered by term frequency across the corpus.
