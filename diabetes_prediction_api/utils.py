import joblib
import matplotlib.pyplot as plt
from sklearn import metrics

from . import config, ROOT_DIR


def get_predictor():
    return joblib.load(f'{ROOT_DIR}/models/{config["model_file_name"]}')


def draw_roc(y_test, predictions, extra_title: str) -> None:
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=0)

    AuROC = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % AuROC)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic [{extra_title}]")
    plt.legend(loc="lower right")
    plt.show()
