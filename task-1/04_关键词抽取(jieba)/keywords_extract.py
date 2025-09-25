# == 利用gensim库调用LSA/LSI和LDA算法实现文档的关键词抽取。 ==

import jieba
import jieba.analyse

def load_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    return text

def extract_keywords(text, topK=20, withWeight=True):   # topK: 提取关键词的个数, withWeight: 是否返回权重
    keywords = jieba.analyse.textrank(text, topK=topK, withWeight=withWeight)
    return keywords

def main():
    path = "../03_关键词抽取(gensim)/article.txt"
    text = load_data(path)

    keywords = extract_keywords(text, topK=10, withWeight=True)

    print("文档关键词：")
    for word, weight in keywords:
        print(f"{word} (权重: {weight:.4f})")

if __name__ == "__main__":
    main()
