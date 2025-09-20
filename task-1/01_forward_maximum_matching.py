def forward_maximum_matching(dic, sentence):
    result = []
    index = 0
    sentence_size = len(sentence)
    while index < sentence_size:
        matched = False
        for sz in range(sentence_size, 0, -1):
            if sz + index > sentence_size: continue
            word = sentence[index:index + sz]
            if word in dic:
                result.append(word)
                index += sz
                matched = True
                break
        if not matched:
            result.append(sentence[index])
            index += 1
    return result

def main():
    sentence = "研究生命起源"
    dictionary = ["研究", "研究生", "生命", "命", "起源"]
    result = forward_maximum_matching(dictionary, sentence)
    print(result)

if __name__ == "__main__":
    main()