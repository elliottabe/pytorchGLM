import torch
import torch.nn as nn
from kornia.geometry.transform import Affine
from torch.nn.modules.activation import Softplus
from torch.nn.modules.linear import Linear
class LinVisNetwork(nn.Module):
    def __init__(self, 
                    in_features, 
                    N_cells, 
                    shift_in=3, 
                    shift_hidden=20,
                    shift_out=3,
                    reg_alph=None, 
                    reg_alphm=None, 
                    reg_laplace=None,
                    lap_M=None,
                    move_features=None, 
                    LinMix=False,
                    train_shifter=False,
                    meanbias = None,
                    device='cuda'):
        super(LinVisNetwork, self).__init__()
        r''' Linear Visual GLM Network
        Args: 
            in_feature: size of the input dimension
            N_cells: the number of cells to fit
            shift_in: size of the input to shifter network. 
            shift_hidden: size of the hidden layer in the shifter network
            shift_out: output dimension of shifter network
            reg_alph: L1 regularization value for visual network
            reg_alphm: L1 regularization value for position network 
            move_feature: the number of position features 
            LinMix: Additive or Multiplicative mixing. LinMix=True for additive, LinMix=False for multiplicative
            train_shifter: Bool for whether training shifter network
            meanbias: can set bias to mean firing rate of each neurons
            device: which device to run network on
        
        
        '''
        self.in_features = in_features
        self.N_cells = N_cells
        self.move_features = move_features
        self.LinMix = LinMix
        self.meanbias = meanbias
        # Cell_NN = {'{}'.format(celln):nn.Sequential(nn.Linear(self.in_features, self.hidden_features),nn.Softplus(),nn.Linear(self.hidden_features, 1)) for celln in range(N_cells)}
        # self.visNN = nn.ModuleDict(Cell_NN)
        self.Cell_NN = nn.Sequential(nn.Linear(self.in_features, self.N_cells,bias=True))
        self.activations = nn.ModuleDict({'SoftPlus':nn.Softplus(),
                                          'ReLU': nn.ReLU(),})
        torch.nn.init.uniform_(self.Cell_NN[0].weight, a=-1e-6, b=1e-6)
        # torch.nn.init.constant_(self.Cell_NN[2].weight, 1)
        # if self.meanbias is not None:
        #     torch.nn.init.constant_(self.Cell_NN[0].bias, meanbias)
        
        # Initialize Regularization parameters
        self.reg_alph = reg_alph
        if self.reg_alph != None:
            self.alpha = reg_alph*torch.ones(1).to(device)
        
        self.reg_laplace = reg_laplace
        if (self.reg_laplace != None) | (lap_M != None):
            self.lalpha = reg_laplace*torch.ones(1).to(device)
            self.lap_M = lap_M.to(device)


        # Initialize Movement parameters
        self.reg_alphm = reg_alphm
        if self.move_features != None:
            if reg_alphm != None:
                self.alpha_m = reg_alphm*torch.ones(1).to(device)
            self.posNN = nn.ModuleDict({'Layer0': nn.Linear(move_features, N_cells)})
            torch.nn.init.uniform_(self.posNN['Layer0'].weight,a=-1e-6,b=1e-6) 
            if self.LinMix==False:
                torch.nn.init.ones_(self.posNN['Layer0'].bias) # Bias = 1 for mult bias=0 for add
            else:    
                torch.nn.init.zeros_(self.posNN['Layer0'].bias) # Bias = 1 for mult bias=0 for add

        # option to train shifter network
        self.train_shifter = train_shifter
        self.shift_in = shift_in
        self.shift_hidden = shift_hidden
        self.shift_out = shift_out
        if train_shifter:
            self.shifter_nn = nn.Sequential(
                nn.Linear(self.shift_in,shift_hidden),
                nn.Softplus(),
                nn.Linear(shift_hidden, shift_out)
            )
        
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight,a=-1e-6,b=1e-6)
            # torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(1e-6)
    
    def forward(self, inputs, move_input=None, eye_input=None, celln=None):
        if self.train_shifter: 
            batchsize, timesize, x, y = inputs.shape
            dxy = self.shifter_nn(eye_input)
            shift = Affine(angle=torch.clamp(dxy[:,-1],min=-30,max=30),translation=torch.clamp(dxy[:,:2],min=-15,max=15))
            inputs = shift(inputs)
            inputs = inputs.reshape(batchsize,-1).contiguous()
        # fowrad pass of GLM 
        x, y = inputs.shape    
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        if celln is not None:
            output = []
            for celln in range(self.N_cells):
                output.append(self.Cell_NN['{}'.format(celln)](inputs))
            output = torch.stack(output).squeeze().T
        else:
            output = self.Cell_NN(inputs)
        # Add Vs. Multiplicative
        if move_input != None:
            if self.LinMix==True:
                output = output + self.posNN['Layer0'](move_input)
            else:
                move_out = self.posNN['Layer0'](move_input)
                # move_out = self.activations['SoftPlus'](move_out)
                # move_out = torch.exp(move_out)
                output = output*move_out
        ret = self.activations['ReLU'](output)
        return ret
    
    def loss(self,Yhat, Y): 
        # if self.LinMix:
        #     loss_vec = torch.mean((Yhat-Y)**2,axis=0)
        # else:
        loss_vec = torch.mean((Yhat-Y)**2,axis=0)
            # loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0)  # Log-likelihood
        if self.move_features != None:
            if self.reg_alph != None:
                l1_reg0 = self.alpha*(torch.linalg.norm(self.Cell_NN[0].weight,axis=1,ord=1))
            else: 
                l1_reg0 = 0
                l1_reg1 = 0
            if self.reg_alphm != None:
                l1_regm = self.alpha_m*(torch.linalg.norm(self.weight[:,-self.move_features:],axis=1,ord=1))
            else: 
                l1_regm = 0
            loss_vec = loss_vec + l1_reg0 + l1_reg1 + l1_regm
        else:
            if self.reg_alph != None:
                l1_reg0 = torch.stack([torch.linalg.vector_norm(NN_params,ord=1) for name, NN_params in self.Cell_NN.named_parameters() if '0.weight' in name])
                loss_vec = loss_vec + self.alpha*(l1_reg0)
            elif self.reg_laplace != None:
                l1_reg2 = torch.mean(torch.matmul(self.Cell_NN[0].weight,torch.matmul(self.lap_M,self.Cell_NN[0].weight.T)),dim=1)
                loss_vec = loss_vec + self.reg_laplace*l1_reg2
        
        return loss_vec
