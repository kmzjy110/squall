import json

with open('../data/squall.json') as f:
    data = json.load(f)

total_alignments =0
nl_overlaps=0
sql_overlaps=0
sql_to_multiple=0
for item in data:
    lim=0
    for alignment_i in item["align"]:
        total_alignments+=1
        for alignment_j in item['align']:
            flag1 = len(set(alignment_i[0]).intersection(set(alignment_j[0])))!=0 and set(alignment_i[0]) !=set(alignment_j[0])

            if flag1:
                nl_overlaps+=1
            flag2 = len(set(alignment_i[1]).intersection(set(alignment_j[1]))) != 0 and set(alignment_i[1]) != set(
                alignment_j[1])
            if flag2:
                sql_overlaps+=1
        if len(alignment_i[1])==1 and len(alignment_i[0])>1:
            if lim<500:
                print(item['sql'][alignment_i[1][0]])
                for ind in alignment_i[0]:
                    print(item['nl'][ind])
                lim +=1
            sql_to_multiple+=1


print("total alignments", total_alignments)
print("nl_overlaps",nl_overlaps)
print("sql_overlaps",sql_overlaps)
print("sql_to_multiple",sql_to_multiple)