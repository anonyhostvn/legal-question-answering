ESP = 1e-10


def calculate_f2i(lis_ground_truth, lis_predict):
    true_predict = len([i for i in lis_predict if i in lis_ground_truth])
    precision = true_predict / (ESP + len(lis_predict))
    recall = true_predict / (ESP + len(lis_ground_truth))
    return (5 * precision * recall) / (4 * precision + recall + ESP)


def calculate_recall(lis_ground_truth, lis_predict):
    true_predict = len([i for i in lis_predict if i in lis_ground_truth])
    return true_predict / (ESP + len(lis_ground_truth))
