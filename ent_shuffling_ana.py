from util import load_file, load_json
import numpy as np



result = load_json('./data/Entity_Shuffling/result.json')
relations = load_json('./data/Entity_Shuffling/relation_label.json')[:len(result)]
rel2id = load_json('./data/relations.json')
id2rel = {v: k for k, v in rel2id.items()}
print(id2rel)

relations = np.array([rel2id[i] for i in relations])

print(len(result))
print(len(relations))


print(relations)
result = np.array(result)

print(np.sum(result==True)/len(result))

# 计算摸个关系的正确率
def f(r, rel):
    p = r[relations==rel]
    if len(p)>50:
        acc = np.sum(p==True)/len(p)
        if acc < 0.8:
            print(id2rel[rel])
            print(acc)

for i in range(410):
    f(result, i)