import torch 
import pandas as pd
from sklearn.model_selection import train_test_split

#let's create a MLP that take 14 inputs and output 2 values 
class EyeMLP(torch.nn.Module):
    def __init__(self):
        super(EyeMLP, self).__init__()
        self.fc1 = torch.nn.Linear(30, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 32)
        self.fc6 = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x
    
def load_data(file_path):
    #load data from a csv file
    data = pd.read_csv(file_path)
    X,y = data.iloc[:, :-2].values, data.iloc[:, -2:].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_model(model_path, device):
    model = EyeMLP()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model