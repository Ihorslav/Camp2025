from collections import defaultdict
pair = []
for i in range(2, 100):
    for j in range(2, 100):
        pair.append([i, j])


def clear(array):
    clean = []
    seen = set()

    for pair1 in array:
        if (pair1[1], pair1[0]) not in seen:
            clean.append(pair1)
            seen.add((pair1[0], pair1[1]))

    for pair1 in clean:
        if (pair1[1] == pair1[0]):
            clean.remove(pair1)
    return clean


def first(array):
    prod_dict = defaultdict(list)
    for pair in array:
        Prod = pair[0]*pair[1]
        prod_dict[Prod].append(pair)
    return prod_dict


cleaned_pairs = clear(pair)
prod_final_dict = defaultdict(list)
prod_dict = first(cleaned_pairs)
for k, v in prod_dict.items():
    if len(prod_dict[k]) > 1:
        prod_final_dict[k] = v

first_pairs = list()

for k, v in prod_final_dict.items():
    for i in v:
        first_pairs.append(i)


def second(array):
    sum_dict = defaultdict(list)
    for pair in array:
        Sum = pair[0]+pair[1]
        sum_dict[Sum].append(pair)
    return sum_dict


second_pairs = second(first_pairs)

cleaned_sum_pairs = clear(pair)
sum_final_dict = defaultdict(list)
sum_dict = second(first_pairs)
for k, v in sum_dict.items():
    if len(sum_dict[k]) > 1:
        sum_final_dict[k] = v

second_final_pairs = list()

for k, v in sum_final_dict.items():
    for i in v:
        second_final_pairs.append(i)

print(second_final_pairs)
