from collections import defaultdict
import math

def load_data(path):
    data = []
    sentences = []
    tags = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line[0])
                tags.append(line[2])
            else:
                data.append(("".join(sentences), tags[:]))
                sentences.clear()
                tags.clear()
        if sentences:        # 文件结尾可能没有空行，防止最后一行数据为存入data
            data.append(("".join(sentences), tags[:]))
    return data

def train_hmm(data):
    states = ['B', 'M', 'E', 'S']
    start_prob = defaultdict(lambda: 1e-6)       # 初始概率
    trans_prob = defaultdict(lambda: defaultdict(lambda: 1e-6))  # 转移概率，lamda匿名函数
    emit_prob = defaultdict(lambda: defaultdict(lambda: 1e-6))   # 发射概率
    state_count = defaultdict(int)      # 每个状态出现次数

    for sentences, tags in data:
        prev_tag = None
        for i, (char, tag) in enumerate(zip(sentences, tags)):
            state_count[tag] += 1
            emit_prob[tag][char] += 1
            if i != 0:
                trans_prob[prev_tag][tag] += 1
            else:
                start_prob[tag] += 1
            prev_tag = tag

    # 转成概率（加对数避免下溢）
    start_prob = {k: math.log(v / len(data)) for k, v in start_prob.items()}    # start_prob.items()返回的是一个「可迭代的键值对序列」。如果没有item，则只会遍历key

    for prev in trans_prob:
        # 计算出第一个状态的值有多少
        total = sum(trans_prob[prev].get(cur, 0) for cur in states)
        total = total + len(states) # 平滑
        # for cur in states:
        #     total += trans_prob[prev].get(cur, 0) + lens(states)
        for cur in trans_prob[prev]:            # 遍历从第一个状态到第二个状态
            count = trans_prob[prev].get(cur, 0) + 1    # 平滑
            trans_prob[prev][cur] = math.log(count / total)

    for tag in emit_prob:
        total = sum(emit_prob[tag].get(cur, 0) for cur in states) + len(states)
        for char in emit_prob[tag]:
            count = emit_prob[tag].get(char) + 1
            emit_prob[tag][char] = math.log(count / total)

    return start_prob, trans_prob, emit_prob, states

def viterbi(sentence, states, start_prob, trans_prob, emit_prob):
    V = [{}]    # V[t][s]: t 时刻状态 s 的最大概率
    path = {}   # path[s]: 到 s 的最优路径

    # for i, data in enumerate(start_prob.items()):
    #     print(type(data), data)

    # 初始化
    for y in states:
        V[0][y] = start_prob.get(y, 1e6) + emit_prob[y].get(sentence[0], 1e6)
        path[y] = [y]

    # 递推
    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}
        for y in states:
            # 选择最大概率的前一个状态
            prob, state = max(
                (V[t-1][y0] + trans_prob[y0].get(y, 1e6) + emit_prob[y].get(sentence[t], 1e6), y0)
                for y0 in states
            )
            V[t][y] = prob
            new_path[y] = path[state] + [y]
        path = new_path

    # 终止：取概率最大的路径
    prob, state = max((V[-1][y], y) for y in states)    # -1表示最后一行
    return path[state]

def cut_sentence(sentence, tags):
    result = []
    word = ""
    for char, tag in zip(sentence, tags):
        word += char
        if tag in ("E", "S"):  # 词结束
            result.append(word)
            word = ""
    if word:
        result.append(word)  # 防止最后没加上
    return result

def main():
    data = load_data("./train.txt")
    # for i,(sentences, tags) in enumerate(data[:10]):
    #     print(f"第{i+1}句：{sentences}，tags为：{tags}")
    start_prob, trans_prob, emit_prob, states = train_hmm(data)

    test_sentence = "中国十四个边境开放城市经济建设成就显著"
    pred_tags = viterbi(test_sentence, states, start_prob, trans_prob, emit_prob)

    print(f"原始句子：{test_sentence}")
    print(f"维特比预测最优隐藏状态序列：{pred_tags}")
    print(f"分词结果：{cut_sentence(test_sentence, pred_tags)}")

if __name__ == '__main__':
    main()
