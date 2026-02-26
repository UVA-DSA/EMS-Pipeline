def rank(scores):
    srt = sorted(scores,key=lambda x: x[1], reverse=True)
    return [i[0] for i in srt],[i[1] for i in srt]
	
def nDCG(rank,gt,group,num):
    from numpy import log2
    for i in group:
        if gt in group[i]:
            break
    ct = len(group[i]) - 1
    idcg = 1.
    idx = 1
    num -= 1
    while (ct&num):
        idcg += 1/log2(idx+2)
        idx += 1
        ct -= 1
        num -= 1
    
    dcg = 0.
    for idx,item in enumerate(rank):
        if item == gt:
            dcg += 2/log2(idx+2)
        elif item in group[i]:
            dcg += 1/log2(idx+2)
        else:
            continue
    return dcg/idcg