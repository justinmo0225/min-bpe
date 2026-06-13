def get_stats(ids):
    stats = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        stats[pair] = stats.get(pair, 0) + 1

    return stats

def merge(ids, pair, new_id):
    merged_stats = []
    i = 0

    while i < len(ids):
        if i == len(ids) - 1:
            merged_stats.append(ids[i])
            break

        if ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged_stats.append(new_id)
            i += 2
        else:
            merged_stats.append(ids[i])
            i += 1

    return merged_stats