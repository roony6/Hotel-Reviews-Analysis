import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stem = PorterStemmer()

# Uncomment these 2 lines to download stopwords - vader_lexicon
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# Main Function
def encode_review(data):
    neg = data['Negative_Review']
    pos = data['Positive_Review']

    print('Cleaning Data...')
    no_neg = 'No Negative'
    no_pos = 'No Positive'
    neg = neg.str.replace(no_neg, '')
    pos = pos.str.replace(no_pos, '')

    print('Tokenizing Data...')
    tokenizer = RegexpTokenizer(r'\w+')
    neg = neg.apply(lambda x: tokenizer.tokenize(x.lower()))
    pos = pos.apply(lambda x: tokenizer.tokenize(x.lower()))

    print('Removing stop words from Data...')
    neg = neg.apply(lambda x: stop_word(x))
    pos = pos.apply(lambda x: stop_word(x))

    print('Stemming Data...')
    neg = neg.apply(lambda x: word_stem(x))
    pos = pos.apply(lambda x: word_stem(x))

    print('Analyzing Data...')
    sia = SentimentIntensityAnalyzer()
    sent_neg = neg.apply(lambda x: sia.polarity_scores(x)['compound'])
    sent_pos = pos.apply(lambda x: sia.polarity_scores(x)['compound'])

    data = pd.concat([data.drop(columns=['Negative_Review', 'Positive_Review']), sent_neg, sent_pos], axis=1)
    return data


# Clearing stop words
def stop_word(text):
    stop = stopwords.words('english')

    words = [x for x in text if x not in stop]
    return words


# Stemming Words
def word_stem(text):
    txt = ' '.join([stem.stem(i) for i in text])
    return txt
