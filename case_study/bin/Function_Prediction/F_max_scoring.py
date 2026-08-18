"""


- The module implements ``scoring_f_max`` method. The method takes a list as input.The List consist of model pipline,real annotation numpy array and protein vector dataframe
The function , call intersection function for true positive and true negative value calculation.It calculates f score.


"""


from imblearn.pipeline import Pipeline

"""def intersection(real_annot, pred_annot):
    count = 0
    tn = 0
    tp = 0
    for i in range(len(real_annot)):
        if real_annot[i] == pred_annot[i]:
            if real_annot[i] == 0:
                tn += 1
            else:
                tp += 1
            count += 1

    return tn, tp"""



"""def scoring_f_max_machine(model_pipline,protein_representation_array,real_annots):

    tn=0
    tp=0
    
    pred_annots=model_pipline.predict(protein_representation_array)
   
    tn,tp=intersection(real_annots, pred_annots)
    fp = list(pred_annots).count(1) - tp
    fn = list(real_annots).count(0) - tn
    recall = tp /(1.0 + (tp + fn))
    precision = tp / (1.0 + (tp + fp))
    f = 0.0
    if precision + recall > 0:
        f = 2 * precision * recall / (precision + recall)
    
    return f"""

# f_max scoring function
import numpy as np

def evaluate_annotation_f_max(real_annots, pred_annots):

    real = np.array(real_annots)
    pred = np.array(pred_annots)

    tp = np.sum((real == 1) & (pred == 1))
    fp = np.sum((real == 0) & (pred == 1))
    fn = np.sum((real == 1) & (pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)



def scoring_f_max_machine(model_pipline, X, y_true):
    """
    CAFA protokolü: predict_proba → threshold sweep → Fmax
    """
    # Probability prediction
    try:
        prob = model_pipline.predict_proba(X)[:, 1]
    except:
        # SVC gibi predict_proba olmayanlar için normalize edilmiş decision score
        decision = model_pipline.decision_function(X)
        prob = (decision - decision.min()) / (decision.max() - decision.min())

    y_true = np.array(y_true)
    
    best_fmax = 0.0
    thresholds = np.linspace(0, 1, 101)

    for th in thresholds:
        y_pred = (prob >= th).astype(int)
        f = evaluate_annotation_f_max(y_true, y_pred)
        print(f)
        if f > best_fmax:
            best_fmax = f

    return best_fmax