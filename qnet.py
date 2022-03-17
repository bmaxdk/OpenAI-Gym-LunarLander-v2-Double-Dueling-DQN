import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        '''
        Builds a feedforward nework with two hidden layers
        Initialize parameters
        
        Params
        =========
        state_size (int): Dimension of each state (input_size)
        action_size (int): dimension of each action (output_size)
        seed (int): Random seed(using 0)
        fc1_units (int): Size of the first hidden layer
        fc2_units (int): Size of the second hidden layer
        '''
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Add the first laer, input to hidden layer
        self.fc1 = nn.Linear(state_size, fc1_units)
        # Add more hidden layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # State-value V
        self.V = nn.Linear(fc2_units, 1)
        
        # Advantage function A
        self.A = nn.Linear(fc2_units, action_size)
        
        
    def forward(self, state):
        """
        Forward pass through the network. Build a network that mps state -> action values.
        
        return Q function
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        
        return V + (A - A.mean(dim=1, keepdim=True))
    
    
# with out Dueling
# class QNetwork(nn.Module):
#     def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
#         '''
#         Builds a feedforward nework with two hidden layers
#         Initialize parameters
        
#         Params
#         =========
#         state_size (int): Dimension of each state (input_size)
#         action_size (int): dimension of each action (output_size)
#         seed (int): Random seed(using 0)
#         fc1_units (int): Size of the first hidden layer
#         fc2_units (int): Size of the second hidden layer
#         '''
#         super(QNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         # Add the first laer, input to hidden layer
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         # Add more hidden layer
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
        
#     def forward(self, state):
#         """
#         Forward pass through the network. Build a network that mps state -> action values.
#         """
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
        
        
        