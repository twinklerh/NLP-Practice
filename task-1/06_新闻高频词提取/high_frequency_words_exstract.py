import jieba
from collections import Counter
import re

# -----------------------------
# 加载文本和停用词
# -----------------------------
def load_data(article_path, stopwords_path=None):
    with open(article_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 按句号、问号、感叹号拆分
    sentences = re.split(r'[.。?？!！]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    # 读取停用词
    stopwords = set()
    if stopwords_path:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
    return sentences, stopwords

# -----------------------------
# 高频词统计
# -----------------------------
def get_top_words(text_list, topK=20, stopwords=None):
    if stopwords is None:
        stopwords = set()
    all_tokens = []
    for text in text_list:
        tokens = jieba.lcut(text)  # 精确分词
        for token in tokens:
            # 去掉空格、标点符号、停用词
            if token.strip() and token not in stopwords and re.match(r'[\u4e00-\u9fff]+|[a-zA-Z]+', token):
                all_tokens.append(token)
    counter = Counter(all_tokens)
    return counter.most_common(topK)

# -----------------------------
# 主函数
# -----------------------------
def main():
    article_path = "./news.txt"
    stopwords_path = "../03_关键词抽取(gensim)/stopwords.txt"

    # 1. 读取文本
    sentences, stopwords = load_data(article_path, stopwords_path)

    # 2. 高频词统计
    top_with_stopwords = get_top_words(sentences, topK=20, stopwords=stopwords)
    top_without_stopwords = get_top_words(sentences, topK=20, stopwords=set())

    # 3. 输出对比
    print("=== 使用停用词表 ===")
    for word, freq in top_with_stopwords:
        print(f"{word}: {freq}")

    print("\n=== 不使用停用词表 ===")
    for word, freq in top_without_stopwords:
        print(f"{word}: {freq}")

if __name__ == "__main__":
    main()
