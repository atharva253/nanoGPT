from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import pandas as pd

def print_results(file):
    df = pd.read_csv(file)
    label = df["lab"]
    pred = df["pred_num"]
    print(f"File: {file}")
    print(f"Accuracy: {accuracy_score(label, pred)}")
    print(f"Precision: {precision_score(label, pred)}")
    print(f"Recall: {recall_score(label, pred)}")
    print(f"F1: {f1_score(label, pred)}")
    print("\n")

print_results("multitask_sentiment_model_1.csv")
print_results("multitask_sentiment_model_2.csv")
