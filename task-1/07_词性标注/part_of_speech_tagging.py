import jieba.posseg as pseg
from snownlp import SnowNLP
import re

def load_stopwords(stopwords_path):
    stopwords = set()
    if stopwords_path:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
    return stopwords

# jieba 词性标注（支持可选停用词过滤）
def pos_tag_jieba(text, remove_stopwords=False, stopwords=None):

    words = pseg.cut(text)
    result = []
    for word, flag in words:
        if remove_stopwords and stopwords and word in stopwords:
            continue
        result.append((word, flag))
    return result


# SnowNLP 词性标注（支持可选停用词过滤）
def pos_tag_snownlp(text, remove_stopwords=False, stopwords=None):
    s = SnowNLP(text)
    words = s.words
    tags = s.tags
    result = []
    for word, tag in zip(words, tags):
        if remove_stopwords and stopwords and word in stopwords:
            continue
        result.append((word, tag))
    return result


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


def main():
    article_path = "../06_新闻高频词提取/news.txt"
    stopwords_path = "../03_关键词抽取(gensim)/stopwords.txt"

    # 1. 读取文本
    with open(article_path, "r", encoding="utf-8") as f:
        text = f.read().replace("　","").strip()

    # 2. 加载停用词
    stopwords = load_stopwords(stopwords_path)

    # 3. 选择是否去掉停用词
    remove_stopwords = True  # 改成 False 就保留停用词

    # 4. jieba 词性标注
    print("=== jieba 词性标注 ===")
    jieba_result = pos_tag_jieba(text, remove_stopwords=remove_stopwords, stopwords=stopwords)
    for word, flag in jieba_result:
        print(f"{word}: {flag}")

    # 5. SnowNLP 词性标注
    print("\n=== SnowNLP 词性标注 ===")
    snownlp_result = pos_tag_snownlp(text, remove_stopwords=remove_stopwords, stopwords=stopwords)
    for word, tag in snownlp_result:
        print(f"{word}: {tag}")

if __name__ == "__main__":
    main()
