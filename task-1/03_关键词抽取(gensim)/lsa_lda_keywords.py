import re
import jieba
from collections import Counter
from gensim import corpora
from gensim.models import LsiModel, LdaModel

def load_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        sentences = f.read()
    article = re.split(r'[.。?？!！]', sentences)
    article = [s.strip() for s in article if s.strip()]
    return article

def tokenize(texts):
    # 用jieba切词，并去掉停用词
    stopwords = set()
    with open("./stopwords.txt", "r", encoding="utf-8") as f:
        for w in f:
            stopwords.add(w.strip())

    tokenized = []
    for sent in texts:
        words = [w for w in jieba.lcut(sent) if w.strip() and w not in stopwords]
        tokenized.append(words)
    return tokenized

def lsa_lsi_extract(corpus, dictionary):
    lsi = LsiModel(corpus=corpus, id2word=dictionary, num_topics=2)
    print("=== LSI 主题 ===")
    for topic in lsi.print_topics():
        print(topic)
    lsi_topic_terms = lsi.show_topic(0, topn=5)
    lsi_keywords = [term for term, weight in lsi_topic_terms]
    print("LSI 关键词：", lsi_keywords)

def lda_extract(corpus, dictionary):
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, random_state=42)
    print("\n=== LDA 主题 ===")
    for topic in lda.print_topics():
        print(topic)
    topic_counter = Counter()
    for doc_bow in corpus:
        for topic_id, prob in lda.get_document_topics(doc_bow):
            topic_counter[topic_id] += prob
    main_topic = topic_counter.most_common(1)[0][0]
    lda_topic_terms = lda.get_topic_terms(main_topic, topn=5)
    lda_keywords = [dictionary[id] for id, prob in lda_topic_terms]
    print("LDA 关键词：", lda_keywords)

def main():
    texts_list = load_data("./article.txt")
    article = tokenize(texts_list)
    print(article)  # 分词结果，便于调试
    dictionary = corpora.Dictionary(article)
    corpus = [dictionary.doc2bow(text) for text in article]
    lsa_lsi_extract(corpus, dictionary)
    lda_extract(corpus, dictionary)

if __name__ == '__main__':
    main()
