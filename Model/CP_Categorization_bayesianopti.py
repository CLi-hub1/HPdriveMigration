import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.model_selection import KFold
import os
import optuna

# Fine-tune distilbert model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data = pd.read_excel(r"filepath\CPannobyhuman.xlsx")
data = data.dropna(subset=["merge"])
texts = data["merge"].tolist()
labels = data[["Technological/Infrastructural", "Institutional", "Behavioral/Cultural", "Nature-Based"]].values

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4, problem_type="multi_label_classification")
model.to(device)

encodings = tokenizer(texts, truncation=True, padding=True)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  # Convert labels to float
        return item

    def __len__(self):
        return len(self.labels)

def get_latest_checkpoint(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = (pred.predictions > 0.5).astype(int) 
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='micro')
    roc_auc_values = []
    for i in range(labels.shape[1]):
        roc_auc = roc_auc_score(labels[:, i], probs[:, i])
        roc_auc_values.append(roc_auc)
    roc_auc = np.mean(roc_auc_values)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

input_ids = np.array(encodings['input_ids'])
attention_mask = np.array(encodings['attention_mask'])

# Objective function to optimize using optuna
def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 5)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [16, 32])
    
    fold_metrics = []
    for train_index, val_index in kf.split(input_ids):
        train_encodings = {
            'input_ids': input_ids[train_index],
            'attention_mask': attention_mask[train_index]
        }
        val_encodings = {
            'input_ids': input_ids[val_index],
            'attention_mask': attention_mask[val_index]
        }
        train_labels, val_labels = labels[train_index], labels[val_index]
        
        train_dataset = MyDataset(train_encodings, train_labels)
        val_dataset = MyDataset(val_encodings, val_labels)
        
        training_args = TrainingArguments(
            output_dir='./resultsBAY',
            evaluation_strategy='epoch',
            logging_dir='./logsBAY',
            save_strategy='epoch',
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        latest_checkpoint = get_latest_checkpoint('./resultsBAY')
        if latest_checkpoint:
            print(f"Loading checkpoint: {latest_checkpoint}")
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            trainer.train()
        
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)
    
    avg_f1 = np.mean([metric['eval_f1'] for metric in fold_metrics])
    
    # Log metrics to trial for later access
    trial.set_user_attr('fold_metrics', fold_metrics)
    
    return avg_f1

# Use optuna for Bayesian optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # Set the maximum number of trials

best_params = study.best_params
best_f1 = study.best_value
best_metrics = study.best_trial.user_attrs['fold_metrics']  # Access the best metrics

print("Best parameters found:", best_params)
print("Best F1 score:", best_f1)
print("Best metrics:", best_metrics)

dfmetrics = pd.DataFrame(best_metrics)
dfparams = pd.DataFrame(best_params, index=['row1'])

dfmetrics.to_excel(r"filepath\metricsbyBAY.xlsx", index=False)
dfparams.to_excel(r"filepath\paramsbyBAY.xlsx", index=False)

# Use the best fine-tuned model to predict new data
def train_best_model(encodings, labels, params):
    dataset = MyDataset(encodings, labels)
    training_args = TrainingArguments(
        output_dir='./best_modelBAY',
        evaluation_strategy='no',
        learning_rate=params['learning_rate'],
        num_train_epochs=params['num_train_epochs'],
        per_device_train_batch_size=params['per_device_train_batch_size']
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    trainer.train()
    return model

best_model = train_best_model(encodings, labels, best_params)

new_data = pd.read_excel(r"filepath\CPpredicted.xlsx")
new_data['Action'] = new_data['Action'].fillna('')
new_data['Action description'] = new_data['Action description'].fillna('')
new_data['merge'] = new_data['Action'] + ". " + new_data['Action description']
new_data = new_data.dropna(subset=["merge"])
new_data = new_data[~(new_data['merge'] == '.')]

new_texts = new_data["merge"].tolist()
new_encodings = tokenizer(new_texts, truncation=True, padding=True)
new_dataset = MyDataset(new_encodings, np.zeros((len(new_texts), 4)))  # Dummy labels
# new_dataset = new_dataset.to(device)

trainer = Trainer(model=best_model)
predictions = trainer.predict(new_dataset)
predicted_labels = (predictions.predictions > 0.5).astype(int)
new_data[['Technological/Infrastructural', 'Institutional', 'Behavioral/Cultural', 'Nature-Based']] = predicted_labels

new_data.to_excel(r"filepath\CPpredictedbyBAY.xlsx", index=False)