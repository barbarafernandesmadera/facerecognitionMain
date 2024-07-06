import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, roc_auc_score,
                             f1_score, accuracy_score,
                             precision_score, recall_score)

def calculate_far_frr(y_true, y_scores, thresholds):
    """
    Calculate the False Acceptance Rate (FAR) and False Rejection Rate (FRR) for given thresholds.

    Parameters:
        y_true (numpy.ndarray): True binary labels.
        y_scores (numpy.ndarray): Predicted scores.
        thresholds (numpy.ndarray): Array of threshold values.

    Returns:
        tuple: Arrays of FAR and FRR values.
    """
    far = []
    frr = []

    for threshold in thresholds:
        y_pred = (y_scores <= threshold).astype(int)
        false_accepts = np.sum((y_pred == 1) & (y_true == 0))
        false_rejects = np.sum((y_pred == 0) & (y_true == 1))
        true_accepts = np.sum((y_pred == 1) & (y_true == 1))
        true_rejects = np.sum((y_pred == 0) & (y_true == 0))

        far.append(false_accepts / (false_accepts + true_rejects))
        frr.append(false_rejects / (false_rejects + true_accepts))

    return np.array(far), np.array(frr)

def calculate_eer(far, frr, thresholds):
    """
    Calculate the Equal Error Rate (EER) and the best threshold.

    Parameters:
        far (numpy.ndarray): Array of FAR values.
        frr (numpy.ndarray): Array of FRR values.
        thresholds (numpy.ndarray): Array of threshold values.

    Returns:
        tuple: EER value, best threshold, and the index of the best threshold.
    """
    diff = np.abs(far - frr)
    min_diff_index = np.argmin(diff)
    eer = (far[min_diff_index] + frr[min_diff_index]) / 2
    best_threshold = thresholds[min_diff_index]
    return eer, np.round(best_threshold, 4), min_diff_index

def evaluate_model(y_true, y_scores, best_threshold):
    """
    Evaluate the model using various metrics.

    Parameters:
        y_true (numpy.ndarray): True binary labels.
        y_scores (numpy.ndarray): Predicted scores.
        best_threshold (float): The threshold for converting scores to binary predictions.

    Returns:
        tuple: AUC-ROC, precision, recall, F1 score, and accuracy.
    """
    y_pred = (y_scores <= best_threshold).astype(int)

    auc_roc = roc_auc_score(y_true, -y_scores)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return auc_roc, precision, recall, f1, accuracy

def plot_roc(y_true, y_scores, image_name='data/results/plot.png'):
    """
    Plot the ROC curve and calculate the AUC.

    Parameters:
        y_true (numpy.ndarray): True binary labels.
        y_scores (numpy.ndarray): Predicted scores.
        image_name (str): Image name to save plot.
    """
    fpr, tpr, thresholds = roc_curve(y_true, -y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='navy', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='#008080', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(image_name, dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_recall_far(far, frr, thresholds,
                    best_threshold, image_name='data/results/plot.png'):
    """
    Plot the FAR and Sensitivity/Recall against thresholds.

    Parameters:
        far (numpy.ndarray): Array of FAR values.
        frr (numpy.ndarray): Array of FRR values.
        thresholds (numpy.ndarray): Array of threshold values.
        best_threshold (float): The best threshold value.
        image_name (str): Image name to save plot.
    """
    sensitivity = 1 - frr

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(thresholds, far, label='FAR', color='#008080')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('FAR', color='#008080')
    ax1.tick_params(axis='y', labelcolor='#008080')

    ax2 = ax1.twinx()
    ax2.plot(thresholds, sensitivity, label='Sensitivity/Recall', color='navy')
    ax2.set_ylabel('Sensitivity/Recall', color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')
    plt.axvline(x=best_threshold, color='darkorange', linestyle='--', label=f'Threshold = {best_threshold}')

    plt.title('FAR and Sensitivity vs Threshold')
    fig.tight_layout()

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    ax1.grid(True)
    plt.savefig(image_name, dpi=300, bbox_inches='tight')

    plt.show()
