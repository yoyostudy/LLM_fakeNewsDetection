import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
import pandas as pd
import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from project_output.model import FinetuneTagger
    
class FinetuneTagger:

    def __init__(
            self,
            train_file,
            basemodel='distilbert-base-uncased',
			epochs=3,  #20
            batch_size=64,  #20
            lr=5e-5
        ):

        self.train_ratio = 0.8
        self.tokenizer = AutoTokenizer.from_pretrained(basemodel)
        self.token_max_len = 64
        self.trainfile = train_file
        self.basemodel = basemodel
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.model = DistilBertForSequenceClassification.from_pretrained(basemodel, num_labels = 2).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-8)
        self.load_training_data()
        

    def load_training_data(self):
        self.training_data = pd.read_csv(self.trainfile, index_col=0).dropna()
        self.testing_data = self.training_data.tail(int(len(self.training_data) * (1-self.train_ratio)))
        self.training_data = self.training_data.head(int(len(self.training_data) * self.train_ratio))
         
    def gen_training_seq(self, data_list):
        sentence_list = self.tokenizer(data_list, is_split_into_words=False, truncation=True, padding='max_length', add_special_tokens=False, max_length=self.token_max_len)
        return sentence_list
    
    def get_training_data(self, start_ind, count):
        batch = []
        for i in range(start_ind, start_ind+count):
            batch.append(self.titles[start_ind+i])
        return batch
    
    def train(self):
        self.model.train()
        loss = float("inf")
        for epoch in range(self.epochs):
            self.training_data = self.training_data.sample(frac=1).reset_index(drop=True)
            self.titles = self.training_data['title'].to_numpy()
            self.label = self.training_data['label'].to_numpy()
            print("current epoch: %d" % (epoch+1))
            with tqdm.tqdm(DataLoader([val for val in zip(self.titles, self.label)], batch_size=self.batch_size)) as train_iter:
                for titles, labels in train_iter:
                    output_batch = []
                    temp_batch = self.gen_training_seq(titles)
                    input_id_batch = temp_batch['input_ids']
                    input_attn_batch = temp_batch['attention_mask']
                    input_id_batch = torch.reshape(torch.LongTensor(input_id_batch).to(device), (len(input_id_batch), self.token_max_len))
                    input_attn_batch = torch.reshape(torch.LongTensor(input_attn_batch).to(device), (len(input_attn_batch), self.token_max_len))
                    output_batch = labels.to(device)
                    self.model.zero_grad()
                    loss = self.model(input_id_batch, attention_mask=input_attn_batch, labels=output_batch).loss

                    loss.backward()
                    self.optimizer.step()
                    train_iter.set_description("loss: %f" % loss)

    def get_acc(self, logits, labels):
        pred_label = np.argmax(logits, axis=1)
        return np.sum(pred_label == labels)


    def validate(self):
        self.model.eval()
        self.titles = self.testing_data['title'].to_numpy()
        self.label = self.testing_data['label'].to_numpy()
        
        total_acc = 0.0
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_auc = 0.0
        num_batches = 0
        all_fpr = []
        all_tpr = []

        with tqdm.tqdm(DataLoader([val for val in zip(self.titles, self.label)], batch_size=self.batch_size)) as train_iter:
            for titles, labels in train_iter:
                temp_batch = self.gen_training_seq(titles)
                input_id_batch = temp_batch['input_ids']
                input_attn_batch = temp_batch['attention_mask']
                input_id_batch = torch.reshape(torch.LongTensor(input_id_batch).to(device), (len(input_id_batch), self.token_max_len))
                input_attn_batch = torch.reshape(torch.LongTensor(input_attn_batch).to(device), (len(input_attn_batch), self.token_max_len))

                logits = self.model(input_id_batch, attention_mask=input_attn_batch)
                logits = logits['logits'].detach().cpu().numpy()
                total_acc += self.get_acc(logits, labels.detach().cpu().numpy())

                # Calculate F1 score, precision, recall
                pred_label = np.argmax(logits, axis=1)
                f1 = f1_score(labels.detach().cpu().numpy(), pred_label, average='macro')
                precision = precision_score(labels.detach().cpu().numpy(), pred_label, average='macro')
                recall = recall_score(labels.detach().cpu().numpy(), pred_label, average='macro')
                fpr, tpr, _ = roc_curve(labels.detach().cpu().numpy(), logits[:, 1])  # Assuming logits[:, 1] are the positive class scores
                auc_score = auc(fpr, tpr)
                
                all_fpr.append(fpr)
                all_tpr.append(tpr)

                total_f1 += f1
                total_precision += precision
                total_recall += recall
                total_auc += auc_score
                num_batches += 1


        # Combine all FPR and TPR values
        all_fpr = np.concatenate(all_fpr)
        all_tpr = np.concatenate(all_tpr)

        # Interpolate TPR and FPR values over a larger range of thresholds
        interp_thresholds = np.linspace(0, 1, 10000)
        interp_tpr = np.interp(interp_thresholds, all_fpr, all_tpr)
        interp_fpr = interp_thresholds
        interp_roc_auc = auc(interp_fpr, interp_tpr)

        print("Validation F1:", total_f1 / num_batches)
        print("Validation Accuracy:", total_acc / len(self.testing_data))
        print("Validation Precision:", total_precision / num_batches)
        print("Validation Recall:", total_recall / num_batches)
        print("Validation AUC:", total_auc / num_batches)


        # Calculate confusion matrix
        cm = confusion_matrix(labels.detach().cpu().numpy(), pred_label)

        # # Plot confusion matrix heatmap
        # plt.figure(figsize=(8, 6))
        # sns.set(font_scale=2)  # Adjust font size for labels
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        #             xticklabels=['Negative', 'Positive'],
        #             yticklabels=['Negative', 'Positive'])
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.title('Confusion Matrix')
        # plt.show()


    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load(self, save_path):
        self.model.load_state_dict(torch.load(save_path))

 

finetuner = FinetuneTagger('data/fake_news_raw.csv')

finetuner.load('model/fine_tuned_model.pt')
#finetuner.train()
#finetuner.save('model/fine_tuned_model.pt')
print("validation")
finetuner.validate()

#finetuner.save('model/fine_tuned_model.pt')
#finetuner.load('model/fine_tuned_model.pt')