# %%
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from torch import nn
import torch.nn.functional as F
from sklearn import preprocessing

torch.manual_seed(42)

data = pd.read_excel(r"C:\Users\YM6018\Desktop\Nanoparctiles-FIB\Raman test\WiTec Raman\20230207\Machine learning\data collection.xlsx")

# Convert condition target to 0 or 1
data["condition"] = (data["condition"] == 'control').astype(float)

# Convert PPy_shell to one-hot encoding
data = pd.get_dummies(data, columns=["PPy_shell"])

# Get target
y = data["condition"].values.reshape(-1,1)

# Remove target and features
del data["condition"]

#Transfer X Y shape from nparray to Tensor
X = torch.from_numpy(data.values.astype(float)).type(torch.float32)
Y = torch.from_numpy(y).type(torch.float32)

# Parameters
lr = 0.0001
loss_fn = nn.BCELoss()
num_of_features = X.shape[1]
batch = 20
epochs = 300

# Build the model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(num_of_features, 1)
        # self.liner_2 = nn.Linear(128, 64)
        # self.liner_3 = nn.Linear(64, 1)
        

    def forward(self, input):
        # x = F.relu(self.liner_1(input))
        # x = F.relu(self.liner_2(x))
        x = F.sigmoid(self.liner_1(input))
        return x

def get_model():
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt

# Calculate accuracy
def accurary(y_pred, y_ture):
    y_pred = (y_pred>= 0.5).type(torch.int32)
    acc = (y_pred == y_ture).float().mean()
    return acc

# Set up k-fold cross-validation

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle= True, random_state=123)

# Train and evaluate the model using k-fold cross-validation
Accurary = []
Loss = [] 

train_loss = []
train_accuracy = []

test_loss = []
test_accuracy = []

fold = 0

for train_idx, test_idx in kf.split(X):
    train_x, test_x = X[train_idx], X[test_idx]
    train_y, test_y = Y[train_idx], Y[test_idx]

    model, optim = get_model()
    
    for epoch in range(epochs):
        for i in range(0, len(train_x), batch):
            batch_size = min(batch, len(train_x) - i)
            x = train_x[i : i + batch_size]
            y = train_y[i : i + batch_size]
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        with torch.no_grad():
            
            epoch_accurary = accurary(model(train_x), train_y) 
            epoch_loss = loss_fn(model(train_x),train_y).data
            train_loss.append(epoch_loss)
            train_accuracy.append(epoch_accurary)
            
            epoch_test_accurary = accurary(model(test_x), test_y) 
            epoch_test_loss = loss_fn(model(test_x),test_y).data 
            test_loss.append(epoch_test_loss)
            test_accuracy.append(epoch_test_accurary)

            print("Fold:", fold+1, "Epoch:", epoch+1, 
                  "Loss:", round(epoch_loss.item(), 3),
                  "Accuracy:", round(epoch_accurary.item(),3),
                  "Test Loss:", round(epoch_test_loss.item(),3),
                  "Test Accuracy:", round(epoch_test_accurary.item(),3))
            
    Accurary.append(round(epoch_accurary.item(),3))
    Loss.append(round(epoch_loss.item(), 3))
    fold += 1


print(train_x)

print("Accuracy:", Accurary)
print("Loss:", Loss)



plt.plot(train_accuracy, label='train')
plt.plot(test_accuracy, label='test')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.title('BCE_Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %%
output = pd.DataFrame()
data_1 = pd.read_excel(r"C:\Users\YM6018\Desktop\Nanoparctiles-FIB\Raman test\WiTec Raman\20230207\Machine learning\data collection.xlsx")

output["columns"] = data_1.columns[0:-2] 
output.loc['1024'] = ['3']
output.loc['1025'] = ['8']
output.loc['1026'] = ['14']
output.loc['1027'] = ['23']

output.set_index("columns", inplace=True)

#%%
Y_array = np.array(Y)
X_array = np.array(X)

Y_int = Y_array.reshape(1,-1).astype(np.int32).flatten()

X_0 = X_array[Y_int == 0]
X_1 = X_array[Y_int == 1]

X_0 = torch.from_numpy(X_0.astype(float)).type(torch.float32)
X_1 = torch.from_numpy(X_1.astype(float)).type(torch.float32)

Grad_list_0 = []
Grad_list_1 = []

for i in range(X_0.shape[0]):
    X_i_sample_0 = X_0[i,:]
    # X_i_sample_0.requires_grad = True
    logits_0 = model.forward(X_i_sample_0)
    Grad_0 = logits_0
    # Grad_0 = torch.autograd.grad(logits_0, X_i_sample_0)
    Grad_num_0 = Grad_0.detach().numpy()
    Grad_list_0.append(Grad_num_0)
    # print(Grad_num)

for i in range(X_1.shape[0]):
    X_i_sample_1 = X_1[i,:]
    # X_i_sample_1.requires_grad = True
    logits_2 = model.forward(X_i_sample_1)
    Grad_1 = logits_2
    # Grad_1 =torch.autograd.grad(logits_2, X_i_sample_1)
    Grad_num_1 = Grad_1.detach().numpy()
    Grad_list_1.append(Grad_num_1)
    

Grad_array_0 = np.array(Grad_list_0)
mean_0 = np.mean(Grad_array_0, axis = 0)
Abs_mean_0 = np.array([abs(x) for x in mean_0])
Abs_mean_0  = torch.from_numpy(Abs_mean_0.astype(float)).type(torch.float32)

Grad_array_1 = np.array(Grad_list_1)
mean_1 = np.mean(Grad_array_1, axis = 0)
Abs_mean_1 = np.array([abs(x) for x in mean_1])
Abs_mean_1  = torch.from_numpy(Abs_mean_1.astype(float)).type(torch.float32)


plt.plot(output.index, mean_0 , "r", lw = 1, label = "0")
plt.xlim(460,2100)
plt.plot(output.index, Abs_mean_1, "b", lw = 1, label = "1")
plt.legend()
plt.show


#%%
Grad_A_list = []

for i in range(X.shape[0]):
    X_A = X[i,:]
    X_A.requires_grad = True
    logits_A = model.forward(X_A)
    Grad_A = torch.autograd.grad(logits_A, X_A)
    Grad_num_A = Grad_A[0].numpy()
    Grad_A_list.append(Grad_num_A)

Grad_array_A = np.array(Grad_A_list)
mean_A = np.mean(Grad_array_A, axis = 0)
Abs_mean_A = np.array([abs(x) for x in mean_A])

X_0_mean = X_0.mean(axis= 0)
X_1_mean = X_1.mean(axis= 0)

X_0_Normal = preprocessing.MinMaxScaler().fit_transform(X_0_mean.reshape(-1,1))
X_1_Normal = preprocessing.MinMaxScaler().fit_transform(X_1_mean.reshape(-1,1))


# Abs_mean_A  = torch.from_numpy(Abs_mean_A.astype(float)).type(torch.float32)
fig, ax = plt.subplots(2, sharex= True)

ax[0].plot(output.index[:-5], X_0_Normal[:-5], "b", lw = 1, label = "0")
ax[0].plot(output.index[:-5], X_1_Normal[:-5], "r", lw = 1, label = "1")
ax[1].plot(output.index[:-5], mean_A[:-5], "g", lw = 1, label = "G")
ax[0].legend()
ax[1].legend()
plt.xlim(460,2100)
#%%
Output_A = pd.DataFrame()
data_A = pd.read_excel(r"C:\Users\YM6018\Desktop\Nanoparctiles-FIB\Raman test\WiTec Raman\20230207\Machine learning\data collection.xlsx")
Output_A["columns"] = data_A.columns[0:-2] 
Output_A.loc['1024'] = ['3']
Output_A.loc['1025'] = ['8']
Output_A.loc['1026'] = ['14']
Output_A.loc['1027'] = ['23']
Output_A["Weight"] = mean_A
Output_A["Abs"] = Abs_mean_A

Output_A.set_index("columns", inplace=True)
weights_dict_A = Output_A["Weight"].to_dict()

Max10_weight_value = Output_A["Weight"][:-5].nlargest(10)
Max10_keys = Output_A["Weight"].nlargest(10).index.tolist()

Min10_weight_value = Output_A["Weight"][:-5].nsmallest(10)
#%%
# #%%

# torch.save(model.state_dict())
# model.load(torch.load("xxx.pt"))

# X_real_test
# Y_rea_test

# pred_output = model(X_real_test).max(-1)[1]
#%%
weights = model.liner_1.weight.detach().numpy()
plt.plot(weights[0])

# %%
weights_1 = weights.argmax()
W = output.index[:-5][weights_1]
plt.boxplot([X_0[:,weights_1], X_1[:,weights_1]])
# %%
