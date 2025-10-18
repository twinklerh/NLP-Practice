import json
import time
import os

def load_data(json_path):
    original_text_list = []
    gold_error_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        for data in data_list:
            original_text_list.append(data['original_text'])
            gold_error_list.append((data['correct_text'], data['wrong_ids']))
    return original_text_list, gold_error_list

def compute_precision(gold_errors, pred_errors):
    """
    计算中文文本纠错的准确率
    gold_errors: 真实错误列表，如 [(start, end, correct)]
    pred_errors: 模型预测错误列表，如 [(start, end, correct)]
    """
    correct = 0
    for gold_error, pre_error in zip(gold_errors, pred_errors):
        gold_text, gold_ids = gold_error
        pred_text, pred_ids = pre_error
        if gold_text == pred_text and set(gold_ids)==set(pred_ids):
            correct = correct + 1

    # 预测总数量
    total_pred = len(pred_errors)
    # print(total_pred)
    if total_pred == 0:
        return 1.0  # 无预测时准确率默认为1（避免除以0）
    return correct / total_pred

def macbert_corrector_predict(original_text_list, gold_error_list):
    from pycorrector import MacBertCorrector
    m = MacBertCorrector("shibing624/macbert4csc-base-chinese")
    start_time = time.time()
    pred_result_list = m.correct_batch(original_text_list)
    end_time = time.time()
    print("运行时间：{}".format(end_time-start_time))
    pred_error_list = []
    for pre_result in pred_result_list:
        temp_ids_list = []
        for error in pre_result['errors']:
            temp_ids_list.append(error[2])
        pred_error_list.append((pre_result['target'], temp_ids_list))

    print("MacBert4CSC模型准确率：{}".format(compute_precision(gold_error_list, pred_error_list)))

def t5_corrector_predict(original_text_list, gold_error_list):
    from pycorrector import T5Corrector
    m = T5Corrector()
    start_time = time.time()
    pred_result_list = m.correct_batch(original_text_list)
    end_time = time.time()
    print("运行时间：{}".format(end_time-start_time))
    pred_error_list = []
    for pre_result in pred_result_list:
        temp_ids_list = []
        for error in pre_result['errors']:
            temp_ids_list.append(error[2])
        pred_error_list.append((pre_result['target'], temp_ids_list))
    print("T5模型准确率：{}".format(compute_precision(gold_error_list, pred_error_list)))

def kemlm_corrector_predict(original_text_list, gold_error_list):
    from pycorrector import Corrector
    m = Corrector(language_model_path='people2014corpus_chars.klm')
    start_time = time.time()
    pred_result_list = m.correct_batch(original_text_list)
    end_time = time.time()
    print("运行时间：{}".format(end_time-start_time))
    pred_error_list = []
    for pre_result in pred_result_list:
        temp_ids_list = []
        for error in pre_result['errors']:
            temp_ids_list.append(error[2])
        pred_error_list.append((pre_result['target'], temp_ids_list))
    print("KenLM模型准确率：{}".format(compute_precision(gold_error_list, pred_error_list)))

def qwen2_7b_instruct_corrector_predict(original_text_list, gold_error_list):
    from pycorrector.gpt.gpt_corrector import GptCorrector
    m = GptCorrector(model_name_or_path="Qwen/Qwen2-7B-Instruct")
    start_time = time.time()
    pred_result_list = m.correct_batch(original_text_list, system_prompt="你是一个中文文本纠错助手。请根据用户提供的原始文本，生成纠正后的文本。")
    end_time = time.time()
    print("运行时间：{}".format(end_time-start_time))
    pred_error_list = []
    for pre_result in pred_result_list:
        temp_ids_list = []
        for error in pre_result['errors']:
            temp_ids_list.append(error[2])
        pred_error_list.append((pre_result['target'], temp_ids_list))
    print("GPT(Qwen2-7B-Instruct)模型准确率：{}".format(compute_precision(gold_error_list, pred_error_list)))

def gpt_1_5b_corrector_predict(original_text_list, gold_error_list):
    from pycorrector.gpt.gpt_corrector import GptCorrector
    m = GptCorrector("shibing624/chinese-text-correction-1.5b")
    start_time = time.time()
    pred_result_list = m.correct_batch(original_text_list, system_prompt="你是一个中文文本纠错助手。请根据用户提供的原始文本，生成纠正后的文本。")
    end_time = time.time()
    print("运行时间：{}".format(end_time-start_time))
    pred_error_list = []
    for pre_result in pred_result_list:
        temp_ids_list = []
        for error in pre_result['errors']:
            temp_ids_list.append(error[2])
        pred_error_list.append((pre_result['target'], temp_ids_list))

    print("GPT(chinese-text-correction-1.5b)模型准确率：{}".format(compute_precision(gold_error_list, pred_error_list)))

def gpt_4b_corrector_predict(original_text_list, gold_error_list):
    from pycorrector.gpt.gpt_corrector import GptCorrector
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # MKL 线程冲突处理
    os.environ["OMP_NUM_THREADS"] = "1"          # 禁止多线程干扰
    m = GptCorrector("twnlp/ChineseErrorCorrector3-4B")
    start_time = time.time()
    pred_result_list = m.correct_batch(original_text_list, system_prompt="你是一个中文文本纠错助手。请根据用户提供的原始文本，生成纠正后的文本。")
    end_time = time.time()
    print("运行时间：{}".format(end_time-start_time))
    pred_error_list = []
    for pre_result in pred_result_list:
        temp_ids_list = []
        for error in pre_result['errors']:
            temp_ids_list.append(error[2])
        pred_error_list.append((pre_result['target'], temp_ids_list))

    print("GPT(ChineseErrorCorrector3-4B)模型准确率：{}".format(compute_precision(gold_error_list, pred_error_list)))

def ernie_csc_corrector_pridict(original_text_list, gold_error_list):
    from pycorrector import ErnieCscCorrector
    m = ErnieCscCorrector(model_name_or_path="ernie-3.0-csc-base")
    start_time = time.time()
    pred_result_list = m.correct_batch(original_text_list)
    end_time = time.time()
    print("运行时间：{}".format(end_time-start_time))
    pred_error_list = []
    for pre_result in pred_result_list:
        temp_ids_list = []
        for error in pre_result['errors']:
            for idx in error['position']:
                temp_ids_list.append(idx)
        pred_error_list.append((pre_result['target'], temp_ids_list))

    print("ErnieCSC模型准确率：{}".format(compute_precision(gold_error_list, pred_error_list)))


def main():
    original_text_list, gold_error_list = load_data("./SIGHAN2015/test.json")
    # macbert_corrector_predict(original_text_list, gold_error_list)
    # t5_corrector_predict(original_text_list, gold_error_list)
    gpt_4b_corrector_predict(original_text_list, gold_error_list)


    #qwen2_7b_instruct_corrector_predict(original_text_list, gold_error_list)

# kemlm_corrector_predict(original_text_list, gold_error_list)
# gpt_1_5b_corrector_predict(original_text_list, gold_error_list)
# ernie_csc_corrector_pridict(original_text_list, gold_error_list)

if __name__ == "__main__":
    main()



# import json
# import os
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"  # 替换为你的HTTP代理
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"  # 替换为你的HTTPS代理
# with open("./ModelScope/test.json", encoding="utf-8") as f:
#     test_datasets_list = json.load(f)
#
# original_text_list = []
# gold_error_list = []
# for test_sample in test_datasets_list:
#     original_text_list.append(test_sample['original_text'])
#     gold_error = (test_sample['correct_text'],test_sample['wrong_ids'])
#     gold_error_list.append(gold_error)

