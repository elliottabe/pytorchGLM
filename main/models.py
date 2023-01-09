import torch
import torch.nn as nn
from kornia.geometry.transform import Affine
from torch.nn.modules.activation import Softplus
from torch.nn.modules.linear import Linear

def model_wrapper(ARGS,**kwargs):
    """Model Wrapper

    Args:
        ARGS (tuple): tuple containing the config dictionary and model class

    Returns:
        model : returns instantiated model of input class. 
    """
    config = ARGS[0]
    Model = ARGS[1]
    model = Model(config['in_features'],config['Ncells'],config,**kwargs)
    return model

class BaseModel(nn.Module):
    def __init__(self, 
                    in_features, 
                    N_cells,
                    config,
                    ):
        super(BaseModel, self).__init__()
        r''' Base GLM Network
        Args: 
            in_feature: size of the input dimension
            N_cells: the number of cells to fit
            config: network configuration file with hyperparameters
        '''
        self.config = config
        self.in_features = in_features
        self.N_cells = N_cells
        
        self.Cell_NN = nn.Sequential(nn.Linear(self.in_features, self.N_cells,bias=True))
        self.activations = nn.ModuleDict({'SoftPlus':nn.Softplus(),
                                          'ReLU': nn.ReLU(),})
        torch.nn.init.uniform_(self.Cell_NN[0].weight, a=-1e-6, b=1e-6)
        
        # Initialize Regularization parameters
        self.L1_alpha = config['L1_alpha']
        if self.L1_alpha != None:
            self.register_buffer('alpha',config['L1_alpha']*torch.ones(1))

      
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight,a=-1e-6,b=1e-6)
            m.bias.data.fill_(1e-6)
        
    def forward(self, inputs, pos_inputs=None):
        output = self.Cell_NN(inputs)
        ret = self.activations['ReLU'](output)
        return ret

    def loss(self,Yhat, Y): 
        loss_vec = torch.mean((Yhat-Y)**2,axis=0)
        if self.L1_alpha != None:
            l1_reg0 = torch.stack([torch.linalg.vector_norm(NN_params,ord=1) for name, NN_params in self.Cell_NN.named_parameters() if '0.weight' in name])
            loss_vec = loss_vec + self.alpha*(l1_reg0)
        return loss_vec


class ShifterNetwork(BaseModel):
    def __init__(self, 
                    in_features, 
                    N_cells, 
                    config,
                    device='cuda'):
        super(ShifterNetwork, self).__init__(in_features, N_cells, config)
        r''' Shifter GLM Network
        Args: 
            in_feature: size of the input dimension
            N_cells: the number of cells to fit
            shift_in: size of the input to shifter network. 
            shift_hidden: size of the hidden layer in the shifter network
            shift_out: output dimension of shifter network
            L1_alpha: L1 regularization value for visual network
            train_shifter: Bool for whether training shifter network
            meanbias: can set bias to mean firing rate of each neurons
        '''
        self.config = config
        ##### shifter network initialization #####
        self.shift_in = config['shift_in']
        self.shift_hidden = config['shift_hidden']
        self.shift_out = config['shift_out']
        self.shifter_nn = nn.Sequential(
                                        nn.Linear(config['shift_in'],config['shift_hidden']),
                                        nn.Softplus(),
                                        nn.Linear(config['shift_hidden'], config['shift_out'])
                                    )


    def forward(self, inputs, shifter_input=None):
        ##### Forward Pass of Shifter #####
        batchsize, timesize, x, y = inputs.shape
        dxy = self.shifter_nn(shifter_input)
        shift = Affine(angle=torch.clamp(dxy[:,-1],min=-45,max=45),translation=torch.clamp(dxy[:,:2],min=-15,max=15))
        inputs = shift(inputs)
        inputs = inputs.reshape(batchsize,-1).contiguous()
        ##### fowrad pass of GLM #####
        x, y = inputs.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = self.Cell_NN(inputs)
        ret = self.activations['ReLU'](output)
        return ret



class MixedNetwork(BaseModel):
    def __init__(self, 
                in_features, 
                N_cells, 
                config,
                device='cuda'):
        super(MixedNetwork, self).__init__(in_features, N_cells, config)
        r''' Mixed GLM Network
        Args: 
            in_feature: size of the input dimension
            N_cells: the number of cells to fit
            L1_alpha: L1 regularization value for visual network
            L1_alpham: L1 regularization value for position network 
            move_feature: the number of position features 
            LinMix: Additive or Multiplicative mixing. LinMix=True for additive, LinMix=False for multiplicative        
        '''
        self.config = config
        self.LinMix = config['LinMix']
        ##### Position Network Initialization #####
        if config['L1_alpham'] != None:
            self.register_buffer('alpha_m',config['L1_alpham']*torch.ones(1))

        self.posNN = nn.Sequential(nn.Linear(config['pos_features'], N_cells))
        torch.nn.init.uniform_(self.posNN[0].weight,a=-1e-6,b=1e-6) 
        if self.LinMix==False:
            torch.nn.init.ones_(self.posNN[0].bias) # Bias = 1 for mult bias=0 for add
        else:    
            torch.nn.init.zeros_(self.posNN[0].bias) # Bias = 1 for mult bias=0 for add

    def forward(self, inputs, pos_inputs):
        x, y = inputs.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = self.Cell_NN(inputs)
        if self.LinMix==True:
            output = output + self.posNN(pos_inputs)
        else:
            move_out = torch.abs(self.posNN(pos_inputs))
            output = output*move_out
        ret = self.activations['ReLU'](output)

    def loss(self,Yhat, Y): 
        loss_vec = torch.mean((Yhat-Y)**2,axis=0)
        if self.L1_alpha != None:
            l1_reg0 = self.alpha*(torch.linalg.norm(self.Cell_NN[0].weight,axis=1,ord=1))
        else: 
            l1_reg0 = 0
            l1_reg1 = 0
        if self.L1_alpham != None:
            l1_regm = self.alpha_m*(torch.linalg.norm(self.posNN[0].weight,axis=1,ord=1))
        else: 
            l1_regm = 0
        loss_vec = loss_vec + l1_reg0 + l1_reg1 + l1_regm
        return loss_vec
