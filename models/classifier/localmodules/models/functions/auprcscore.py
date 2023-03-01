from sklearn.metrics import auc, precision_recall_curve

def auprc_score(y_true, y_pred):
    """
    Calcula a área de baixo da curva do gráfico precision/recall.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc_score = auc(recall, precision)
    return auprc_score