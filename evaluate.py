import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

#evaluate model performance on test set
#calculates accuracy, precision, recall with weighted averaging for multiclass
#original function from xai grok llm
def evaluate_model(model: Sequential, xtest: np.ndarray, ytest: np.ndarray, 
                   model_name: str = "model") -> tuple:
    
    ypred = model.predict(xtest)
    ypred_classes = np.argmax(ypred, axis=1)
    ytest_classes = np.argmax(ytest, axis=1)
    
    accuracy = accuracy_score(ytest_classes, ypred_classes)
    precision = precision_score(ytest_classes, ypred_classes, average='weighted')
    recall = recall_score(ytest_classes, ypred_classes, average='weighted')
    
    print(f"\n{model_name} evaluation:")
    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    
    return accuracy, precision, recall

#generate detailed classification report for debugging misclassifications
#maps numeric labels back to composer names for interpretability
def detailed_evaluation(model: Sequential, xtest: np.ndarray, ytest: np.ndarray):
    ypred = model.predict(xtest)
    ypred_classes = np.argmax(ypred, axis=1)
    ytest_classes = np.argmax(ytest, axis=1)
    
    composer_names = ['bach', 'beethoven', 'chopin', 'mozart']
    
    print("\ndetailed classification report:")
    print(classification_report(ytest_classes, ypred_classes, 
                                 target_names=composer_names))
    
    return ypred_classes, ytest_classes

#compare multiple model performances side by side
#useful for tracking improvements across model versions
def compare_models(results: dict):
    print("\nmodel comparison:")
    print("-" * 60)
    print(f"{'model':<20} {'accuracy':<12} {'precision':<12} {'recall':<12}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        acc, prec, rec = metrics
        print(f"{model_name:<20} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f}")
