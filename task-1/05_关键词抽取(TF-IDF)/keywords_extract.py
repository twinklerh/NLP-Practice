import jieba, math, re
from collections import Counter

def load_data(stopwords_path, article_path):
    stopwords = set()
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    with open(article_path, 'r', encoding="utf-8") as f:
        sentences = f.read()
    article = re.split(r'[.。?？!！]', sentences)
    article = [s.strip() for s in article if s.strip()]
    return stopwords, article

# 清洗分词结果：保留中文和英文单词，去掉停用词和标点
def clean_tokens(tokens, stopwords):
    cleaned = []
    for token in tokens:
        # 保留中文或英文单词
        if (re.match(r'[\u4e00-\u9fff]+', token) or re.match(r'[a-zA-Z]+', token)) and token not in stopwords:
            cleaned.append(token)
    return cleaned

def compute_tfidf(texts, topK=10, stopwords=set()):
    docs_tokens = []
    for text in texts:
        tokens = list(jieba.cut(text))
        tokens = clean_tokens(tokens, stopwords)
        docs_tokens.append(tokens)

    # 统计 TF
    tf_list = []
    for tokens in docs_tokens:
        counts = Counter(tokens)
        total = sum(counts.values())
        tf = {word: count / total for word, count in counts.items()}
        tf_list.append(tf)

    # 统计 DF
    df = Counter()
    for tokens in docs_tokens:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df[token] += 1

    # 计算 IDF
    N = len(docs_tokens)
    idf = {word: math.log(N / (df[word] + 1)) + 1 for word in df}

    # 计算 TF-IDF (只对第一个文档)
    tfidf = {word: tf_list[0][word] * idf[word] for word in tf_list[0]}
    tfidf_sorted = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)

    return tfidf_sorted[:topK]

def main():
    stopwords, article_list = load_data("../03_关键词抽取(gensim)/stopwords.txt",
                                        "../03_关键词抽取(gensim)/article.txt")
    article_text = " ".join(article_list)

    keywords = compute_tfidf([article_text], topK=10, stopwords=stopwords)

    print("文档关键词：")
    for word, weight in keywords:
        print(f"{word} (权重: {weight:.4f})")

if __name__ == "__main__":
    main()
