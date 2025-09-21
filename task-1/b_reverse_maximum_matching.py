
def reverse_maximum_matching(dictionary, sentence):
    sentence_length = len(sentence)
    result = []
    index = sentence_length
    while index > 0:
        matched = False
        for sz in range(sentence_length, 0, -1):
            word = sentence[index - sz: index]
            if word in dictionary:
                result.append(word)
                matched = True
                index = index - sz
                break
        if not matched:
            result.append(sentence[index])
            index -= 1
    return result

def main():
    sentence = "研究生命起源"
    dictionary = ["研究", "研究生", "生命", "命", "起源"]
    result = reverse_maximum_matching(dictionary, sentence)
    print(result)

if __name__ == "__main__":
    main()