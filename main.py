import wikipediaapi
import re
from razdel import sentenize
from razdel import tokenize
import pymorphy3 as pm
import numpy as np
import os.path

wiki = wikipediaapi.Wikipedia("Information Retrieval", 'ru')
morph = pm.MorphAnalyzer()

mode = int(input('Enter 1 if you want to add all possible lemmas to collection, else enter 0:\n'))

titles = [
    "Абхазские_пираты",
    "Абхазские_негры",
    "Гуам_(военно-морская_база)",
    "Суперцветение"
]

# getting texts and saving them to files
collection_text = []
for title in titles:
    filename = f"{title}.txt"

    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
            collection_text.append(text)
    else:
        text = wiki.page(title).text if wiki.page(title).exists() else ""
        collection_text.append(text)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)

# text to sentences
collection_doc = []
for text in collection_text:
    sentences = list(sentenize(text))
    for sentence in sentences:
        collection_doc.append(sentence.text)


# transforming to lemmas
if mode == 1:
    collection_morph = \
        [[[lemma.normal_form for lemma in morph.parse(word)]
          for word in map(lambda x: x.text, tokenize(sentence))] for sentence in collection_doc]

    collection_tmp = []
    for item in collection_morph:
        buffer = []
        for sentence in item:
            for lemma in sentence:
                buffer.append(lemma)
        collection_tmp.append(buffer)
    collection_morph = collection_tmp
else:
    collection_morph = \
        [[morph.parse(word)[0].normal_form
          for word in map(lambda x: x.text, tokenize(sentence))] for sentence in collection_doc]

# clearing collection
collection_sent = []
collection_cleared = []
unique_lemmas = []
doc_index = 0

for doc in collection_morph:
    sentence_cleared = []
    for lemma in doc:
        word = re.fullmatch('[а-яА-ЯёЁ]+(-[а-яА-ЯёЁ]+)?', lemma)
        if word:
            w = word.group(0)
            sentence_cleared.append(w)
            unique_lemmas.append(w)
    if sentence_cleared:
        collection_sent.append(collection_doc[doc_index])
        collection_cleared.append(sentence_cleared)
    doc_index += 1

unique_lemmas = list(dict.fromkeys(unique_lemmas))
ind_lemmas = {word: i for i, word in enumerate(unique_lemmas)}

# Calculating tf
tf = np.zeros((len(collection_cleared), len(unique_lemmas)))
for idx, doc in enumerate(collection_cleared):
    for word in doc:
        tf[idx][ind_lemmas[word]] += 1

# Calculating df
df = np.zeros(len(unique_lemmas))
for idx, doc in enumerate(collection_cleared):
    for word in set(doc):
        df[ind_lemmas[word]] += 1

# Calculating idf
idf = np.log10(len(collection_cleared) / df)

# calculating tf-idf vectors
tf_idf = tf * idf

# calculating tf vector model
tf_only = np.copy(tf)

# User queries - wikipedia facts
queries = ['Суперцветение видно даже из космоса.',
           'Военно-морская база вскоре после открытия стала известна как «Тихоокеанский супермаркет».',
           'Пираты и негры — это часть истории Абхазии']

for query in queries:

    # query to lemmas
    if mode == 1:
        query_morph = \
            [[lemma.normal_form for lemma in morph.parse(word)]
             for word in map(lambda x: x.text, tokenize(query))]

        query_tmp = []
        for words in query_morph:
            for lemma in words:
                query_tmp.append(lemma)
        query_morph = query_tmp
    else:
        query_morph = \
            [morph.parse(word)[0].normal_form
             for word in map(lambda x: x.text, tokenize(query))]

    query_cleared = [word.group(0)
                     for word in map(lambda x: re.fullmatch('[а-яА-ЯёЁ]+(-[а-яА-ЯёЁ]+)?', x), query_morph)
                     if word]

    if not query_cleared:
        print('No words!!!')
        exit()

    collection_words = set(query_cleared) & set(unique_lemmas)

    if not collection_words:
        print(f'Words are not in the collection!!!!!!!!!!')
        exit()

    tf_query = np.zeros(len(unique_lemmas))
    tf_idf_query = np.zeros(len(unique_lemmas))
    for i in collection_words:
        tf_query[ind_lemmas[i]] = query_cleared.count(i)
        tf_idf_query[ind_lemmas[i]] = query_cleared.count(i) * idf[ind_lemmas[i]]

    # Cosine similarity
    similarity_idf = [np.dot(doc, tf_idf_query) / (np.linalg.norm(doc) * np.linalg.norm(tf_idf_query))
                      for doc in tf_idf]
    max_index_idf = np.argmax(similarity_idf)
    top_5_idf = sorted(range(len(similarity_idf)), key=lambda i: similarity_idf[i], reverse=True)[:5]
    top_5_values_idf = [(collection_sent[i], similarity_idf[i]) for i in top_5_idf]

    print(
        f'Model: Tf-idf\nDocument number: {max_index_idf}\n'
        f'Search result: {collection_sent[max_index_idf]}\nQuery: {query}\n'
    )
    print('Top 5 relevant documents: ')
    for doc, value in top_5_values_idf:
        print(f'Document: {doc}\nValue: {value}\n')
    print()
    # Cosine similarity
    similarity_tf = [np.dot(doc, tf_query) / (np.linalg.norm(doc) * np.linalg.norm(tf_query))
                     for doc in tf_only]
    max_index_tf = np.argmax(similarity_tf)
    top_5_tf = sorted(range(len(similarity_tf)), key=lambda i: similarity_tf[i], reverse=True)[:5]
    top_5_values_tf= [(collection_sent[i], similarity_tf[i]) for i in top_5_tf]

    print(
        f'Model: Tf\nDocument number: {max_index_tf}\n'
        f'Search result: {collection_sent[max_index_tf]}\nQuery: {query}\n'
    )
    print('Top 5 relevant documents: ')
    for doc, value in top_5_values_tf:
        print(f'Document: {doc}\nValue: {value}\n')
    print()
    # p(q | d) = (for t in q) П((1 - ld) * p(t) + ld * p(t | M)) - formula
    lds = [0.5, 0.9]
    for ld in lds:
        word_freq = np.zeros(len(unique_lemmas))
        words_num = sum(map(len, collection_cleared))

        for elem in unique_lemmas:
            count = 0
            for doc in collection_cleared:
                count += doc.count(elem)
            word_freq[ind_lemmas[elem]] = count
        word_freq = word_freq / words_num
        products = []
        for doc in collection_cleared:
            p = 1
            for word in collection_words:
                word_count = doc.count(word) / len(doc)
                p *= ld * word_count + word_freq[ind_lemmas[word]] * (1 - ld)
            products.append(p)

        max_index_prob = np.argmax(products)
        top_5_p = sorted(range(len(products)), key=lambda i: products[i], reverse=True)[:5]
        top_5_values_p = [(collection_sent[i], products[i]) for i in top_5_p]

        print(f'Model: Language probability model\nDocument number: {max_index_prob}\n'
              f'Search result: {collection_sent[max_index_prob]}\n'
              f'Query: {query}\nLambda: {ld}\nProbability: {products[max_index_prob]}\n'
              )
        print('Top 5 relevant documents: ')
        for doc, value in top_5_values_p:
            print(f'Document: {doc}\nValue: {value}\n')
        print()
    print()
    print('*' * 120)
    print()
    print()
