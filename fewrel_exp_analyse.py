from util import load_file, load_json
import numpy as np

exp_num = 18
epoch_num = 4

gold = list(open(f'./output_fewrel_ke_{exp_num}/test_gold_{epoch_num}.txt', 'r'))
pred = list(open(f'./output_fewrel_ke_{exp_num}/test_pred_{epoch_num}.txt', 'r'))
ka = list(open('./data/fewrel/ka_dev.txt'))
kc = list(open('./data/fewrel/k_correct.txt'))
idx2rel = load_json('./data/fewrel/idx2label')
rel2num = load_json('./rel2num.json')


def exp_res():
    matrix = np.zeros((80, 80))
    for i in range(len(gold)):
        g = int(gold[i].strip())
        p = int(pred[i].strip())

        matrix[g, p] += 1

    print(matrix)

    gold_sum = np.sum(matrix, axis=1)
    pred_sum = np.sum(matrix, axis=0)

    # print(gold_sum)
    # print(pred_sum)
    idx2rel = load_json('./data/fewrel/idx2label')
    rel2num = load_json('./rel2num.json')

    for i in range(80):
        p = matrix[i, i]/pred_sum[i]*100
        r = matrix[i, i]/gold_sum[i]*100
        num = rel2num.get(idx2rel[str(i)], 0)

        if p < 50 or r < 50:
            print(f'class:{i}  p:{p:.2f}%, r:{r:.2f}%, rel_num:{num}\n')



def cal_acc_condition(express):
    matrix = np.zeros((80, 80))
    num = 0
    for i in range(len(gold)):
        g = int(gold[i].strip())
        p = int(pred[i].strip())
        k = int(ka[i].strip())
        kt = int(kc[i].strip())

        if idx2rel[str(g)] not in rel2num.keys():
            # print('pass:', idx2rel[str(g)])
            continue

        if eval(express):
            matrix[g, p] += 1
            num += 1

    true = 0
    sum = np.sum(matrix)
    for i in range(80):
        true += matrix[i, i]
    print(f'ACC:{true / sum * 100:.2f}%')
    print(f'Ratio:{num / len(gold) * 100:.2f}%')


def f():
    s0 = 'kt == 1'
    s1 = 'kt == 1 and k==11'
    s2 = 'kt == 1 and k==12'
    s3 = 'kt == 0'
    s4 = 'kt == 0 and k==0'
    s5 = 'kt == 0 and k==1'
    s6 = 'kt == 0 and k==2'
    s7 = 'kt == 0 and k==11'
    s8 = 'kt == 0 and k==12'

    prompt = ['']
    for i in range(9):
        cal_acc_condition(eval(f's{i}'))
        print()

# cal_acc_condition()
f()
# 情况1：包含正确答案(88.34%)
#     ACC：91.22%
#     （1）一个三元组包含文本中头尾实体(69.09%)
#         ACC：93.73%
#     （2）两个三元组包含文本中头尾实体(19.24%)
#         ACC：82.23%
#
# 情况2：没有正确答案(9.16%)
#     ACC：65.84%
#     （1）没有噪音(0.59%)
#         ACC：68.92%
#     （2）一格噪音(4.77%)
#         ACC：72.00%
#     （3）二格噪音(2.64%)
#         ACC：57.91.41%
#     （4）一个正确头尾实体(1.12%)
#         ACC：64.44%
#     （5）两个正确头尾实体(0.05%)
#         ACC：12.50%

def ana_knowledge():
    matrix_with_k = np.zeros((80, 80))
    matrix_with_knk = np.zeros((80, 80))
    matrix_with_n = np.zeros((80, 80))
    matrix_without_k = np.zeros((80, 80))
    for i in range(len(gold)):
        g = int(gold[i].strip())
        p = int(pred[i].strip())
        k = int(ka[i].strip())

        if idx2rel[str(g)] not in rel2num.keys():
            # print('pass:', g)
            continue

        if k == 11:
            matrix_with_k[g, p] += 1
        elif k > 11:
            matrix_with_knk[g, p] += 1

        elif k == 0:
            matrix_without_k[g, p] += 1
        else:
            matrix_with_n[g, p] += 1

    # matrix = matrix_without_k
    matrix = matrix_with_n+matrix_without_k+matrix_with_k


    true = 0
    t_wk = 0
    t_wok = 0
    t_wn = 0
    t_knk = 0
    sum = np.sum(matrix)
    sum_wk = np.sum(matrix_with_k)
    sum_wok = np.sum(matrix_without_k)
    sum_wn = np.sum(matrix_with_n)
    sum_knk = np.sum(matrix_with_knk)
    for i in range(80):
        true += matrix[i, i]
        t_wok += matrix_without_k[i, i]
        t_wn += matrix_with_n[i, i]
        t_wk += matrix_with_k[i, i]
        t_knk += matrix_with_knk[i, i]
    print(f'ACC:{true/sum*100:.2f}%')
    print(f'ACC w knowledge:{t_wk/sum_wk*100:.2f}%')
    print(f'ACC w knk:{t_knk/sum_knk*100:.2f}%')
    print(f'ACC w/o knowledge:{t_wok/sum_wok*100:.2f}%')
    print(f'ACC w noise:{t_wn/sum_wn*100:.2f}%')

    gold_sum = np.sum(matrix_with_k, axis=1)
    for i in range(80):
        # p = matrix[i, i]/pred_sum[i]*100
        r = matrix_with_k[i, i]/gold_sum[i]*100
        num = rel2num.get(idx2rel[str(i)], 0)

        # if r < 50:
        print(gold_sum[i])
        print(f'class:{i}  r:{r:.2f}%, rel_num:{num}\n')




# ana_knowledge()