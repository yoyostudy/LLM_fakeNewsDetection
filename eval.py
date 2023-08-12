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

from model import FinetuneTagger
    

finetuner = FinetuneTagger()

finetuner.load('model/fine_tuned_model.pt')
#finetuner.train()
#finetuner.save('model/fine_tuned_model.pt')
print("validation")
finetuner.validate()
