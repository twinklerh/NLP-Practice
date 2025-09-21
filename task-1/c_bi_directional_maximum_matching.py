from a_forward_maximum_matching import forward_maximum_matching
from b_reverse_maximum_matching import reverse_maximum_matching

def bi_directional_maximum_matching(dictionary, sentence):
    result1 = forward_maximum_matching(dictionary, sentence)
    result2 = reverse_maximum_matching(dictionary, sentence)

    # print(str("fmm_result:"+str(result1)))
    # print(str("rmm_result:"+str(result2)))

     # 规则：词数少优先，单字词少优先
    if len(result1) < len(result2):
        return result1
    elif len(result1) > len(result2):
        return result2
    else:
        fmm_single = sum(1 for w in result1 if len(w) == 1)
        rmm_single = sum(1 for w in result2 if len(w) == 1)
        # print(str("fmm_single:"+str(fmm_single)))
        # print(str("rmm_single:"+str(rmm_single)))
        if fmm_single >= rmm_single:
            return result1
        return result2


def main():
    sentence = "研究生命起源"
    dictionary = ["研究", "研究生", "生命", "命", "起源"]
    result = bi_directional_maximum_matching(dictionary, sentence)
    print(str("result:"+str(result)))

if __name__ == "__main__":
    main()