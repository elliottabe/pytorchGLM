import torch
import torch.nn as nn
from kornia.geometry.transform import Affine
from torch.nn.modules.activation import Softplus
from torch.nn.modules.linear import Linear
# torch.backends.cudnn.benchmark = True
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoissonGLM_AddMult(nn.Module):
    def __init__(self, 
                    in_features, 
                    out_features, 
                    shift_in=3, 
                    shift_hidden=20,
                    shift_out=3,
                    hidden_move=15,
                    bias=True,
                    reg_alph=None, 
                    reg_alphm=None, 
                    move_features=None, 
                    meanfr=None, 
                    init_sta=None, 
                    LinMix=False,
                    LinNonLinMix=False,
                    NonLinLayer=False,
                    train_shifter=False,
                    device='cuda'):
        super(PoissonGLM_AddMult, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.move_features = move_features
        self.LinMix = LinMix
        self.LinNonLinMix = LinNonLinMix
        self.NonLinLayer=NonLinLayer
        # Initialize weights and bias
        if init_sta is not None:
            self.weight = torch.nn.Parameter(init_sta, requires_grad=True)
            self.init_sta = True
        else:
            self.init_sta = False
            self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features),)
        
        if bias:
            if meanfr is not None:
                self.bias = torch.nn.Parameter(meanfr,requires_grad=True)
                self.meanfr = True
            else:
                self.meanfr = None
                self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize Regularization parameters
        self.reg_alph = reg_alph
        if self.reg_alph != None:
            self.alpha = reg_alph*torch.ones(out_features).to(device)

        # Initialize Movement parameters
        self.reg_alphm = reg_alphm
        if self.move_features != None:
            if reg_alphm is not None:
                self.alpha_m = reg_alphm*torch.ones(out_features).to(device)
            if self.NonLinLayer:
                self.NonLinMixLayer = nn.Sequential(nn.Linear(move_features,hidden_move), 
                                                    nn.Softplus(),
                                                    nn.Linear(hidden_move,out_features))
                # self.move_weights = nn.Parameter(1e-6*torch.ones(out_features,1), requires_grad=True)
            elif self.LinNonLinMix:
                self.moveW_mul = nn.Parameter(torch.zeros(out_features,move_features), requires_grad=True)
                self.moveW_add = nn.Parameter(torch.zeros(out_features,move_features), requires_grad=True)
                self.biasm_mul = torch.nn.Parameter(torch.zeros(out_features),requires_grad=True)
                self.biasm_add = torch.nn.Parameter(torch.zeros(out_features),requires_grad=True)
                self.gamma = torch.nn.Parameter(.5*torch.ones(out_features),requires_grad=True)
            else:
                self.move_weights = nn.Parameter(torch.zeros(out_features,move_features), requires_grad=True)
                self.bias_m = torch.nn.Parameter(torch.zeros(out_features),requires_grad=True)

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
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.NonLinLayer:
            torch.nn.init.constant_(self.NonLinMixLayer[0].weight,1e-6)
            torch.nn.init.zeros_(self.NonLinMixLayer[0].bias)
            torch.nn.init.constant_(self.NonLinMixLayer[2].weight,1e-6)
            torch.nn.init.zeros_(self.NonLinMixLayer[2].bias)
        if self.init_sta == False:
            torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            if self.meanfr == None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / torch.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, move_input=None, eye_input=None):
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
        output = inputs.matmul(self.weight.T)
        if self.bias is not None:
            output = output + self.bias

        # Add Vs. Multiplicative
        if move_input != None:
            if self.LinMix==True:
                output = output + (move_input.matmul(self.move_weights.T)+ self.bias_m)
            elif self.LinNonLinMix:
                output = (1-self.gamma)*output*torch.exp(move_input.matmul(self.moveW_mul.T) + self.biasm_mul) + self.gamma*(output+(move_input.matmul(self.moveW_add.T)+ self.biasm_add))
            else:
                if self.NonLinLayer:
                    move_out = self.NonLinMixLayer(move_input)
                else: 
                    move_out = torch.exp(move_input.matmul(self.move_weights.T) + self.bias_m)
                output = output*move_out
        ret = torch.relu(output)
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def loss(self,Yhat, Y): 
        if self.LinMix:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
        else:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
            # loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0)  # Log-likelihood
        if self.move_features != None:
            if self.reg_alph != None:
                l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
            else: 
                l1_reg = 0
            if self.reg_alphm != None:
                l1_regm = self.alpha_m*(torch.linalg.norm(self.weight[:,-self.move_features:],axis=1,ord=1))
            else: 
                l1_regm = 0
            loss_vec = loss_vec + l1_reg + l1_regm
        else:
            if self.reg_alph != None:
                l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
                loss_vec = loss_vec + l1_reg
        
        return loss_vec



# class VisNetwork(nn.Module):
#     def __init__(self, 
#                     in_features, 
#                     hidden_features,
#                     N_cells, 
#                     shift_in=3, 
#                     shift_hidden=20,
#                     shift_out=3,
#                     hidden_move=15,
#                     reg_alph=None, 
#                     reg_alphm=None, 
#                     move_features=None, 
#                     LinMix=False,
#                     train_shifter=False,
#                     device='cuda'):
#         super(VisNetwork, self).__init__()

#         self.in_features = in_features
#         self.hidden_features = hidden_features
#         self.N_cells = N_cells
#         self.move_features = move_features
#         self.LinMix = LinMix

#         Cell_NN = {'{}'.format(celln):nn.Sequential(nn.Linear(self.in_features, self.hidden_features),nn.Softplus(),nn.Linear(self.hidden_features, 1)) for celln in range(N_cells)}
#         self.visNN = nn.ModuleDict(Cell_NN)
#         self.activations = nn.ModuleDict({'SoftPlus':nn.Softplus(),
#                                           'ReLU': nn.ReLU(),
#         })
#         # Initialize Regularization parameters
#         self.reg_alph = reg_alph
#         if self.reg_alph != None:
#             self.alpha = reg_alph*torch.ones(1).to(device)

#         # Initialize Movement parameters
#         self.reg_alphm = reg_alphm
#         if self.move_features != None:
#             if reg_alphm != None:
#                 self.alpha_m = reg_alphm*torch.ones(1).to(device)
#             self.posNN = nn.ModuleDict({'Layer0': nn.Linear(move_features, 1)})

#         # option to train shifter network
#         self.train_shifter = train_shifter
#         self.shift_in = shift_in
#         self.shift_hidden = shift_hidden
#         self.shift_out = shift_out
#         if train_shifter:
#             self.shifter_nn = nn.Sequential(
#                 nn.Linear(self.shift_in,shift_hidden),
#                 nn.Softplus(),
#                 nn.Linear(shift_hidden, shift_out)
#             )
  
#     def forward(self, inputs, move_input=None, eye_input=None):
#         if self.train_shifter: 
#             batchsize, timesize, x, y = inputs.shape
#             dxy = self.shifter_nn(eye_input)
#             shift = Affine(angle=dxy[:,-1],translation=dxy[:,:2])
#             inputs = shift(inputs)
#             inputs = inputs.reshape(batchsize,-1).contiguous()
#         # fowrad pass of GLM 
#         x, y = inputs.shape    
#         if y != self.in_features:
#             print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
#             return 0
        
#         output = []
#         for celln in range(self.N_cells):
#             output.append(self.visNN['{}'.format(celln)](inputs))
#         output = torch.stack(output).squeeze().T

#         # Add Vs. Multiplicative
#         if move_input != None:
#             if self.LinMix==True:
#                 output = output + self.posNN['Layer0'](move_input)
#             else:
#                 move_out = self.posNN['Layer0'](move_input)
#                 # move_out = self.activations['SoftPlus'](move_out)
#                 move_out = torch.exp(move_out)
#                 output = output*move_out
#         ret = self.activations['ReLU'](output)
#         return ret
    
#     def loss(self,Yhat, Y): 
#         if self.LinMix:
#             loss_vec = torch.mean((Yhat-Y)**2,axis=0)
#         else:
#             loss_vec = torch.mean((Yhat-Y)**2,axis=0)
#             # loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0)  # Log-likelihood
#         if self.move_features != None:
#             if self.reg_alph != None:
#                 l1_reg0 = self.alpha0*(torch.linalg.norm(self.visNN['Layer0'].weight,axis=1,ord=1))
#                 l1_reg1 = self.alpha1*(torch.linalg.norm(self.visNN['Layer1'].weight,axis=1,ord=1))
#             else: 
#                 l1_reg0 = 0
#                 l1_reg1 = 0
#             if self.reg_alphm != None:
#                 l1_regm = self.alpha_m*(torch.linalg.norm(self.weight[:,-self.move_features:],axis=1,ord=1))
#             else: 
#                 l1_regm = 0
#             loss_vec = loss_vec + l1_reg0 + l1_reg1 + l1_regm
#         else:
#             if self.reg_alph != None:
#                 l1_reg0 = torch.stack([torch.linalg.vector_norm(NN_params,ord=1) for name, NN_params in visNN.named_parameters() if '0.weight' in name])
#                 l1_reg2 = torch.stack([torch.linalg.vector_norm(NN_params,ord=1) for name, NN_params in visNN.named_parameters() if '2.weight' in name])
#                 loss_vec = loss_vec + self.alpha*(l1_reg0 +l1_reg2)
        
#         return loss_vec


class VisNetwork(nn.Module):
    def __init__(self, 
                    in_features, 
                    hidden_features,
                    N_cells, 
                    shift_in=3, 
                    shift_hidden=20,
                    shift_out=3,
                    hidden_move=15,
                    reg_alph=None, 
                    reg_alphm=None, 
                    move_features=None, 
                    LinMix=False,
                    train_shifter=False,
                    meanbias = None,
                    device='cuda'):
        super(VisNetwork, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.N_cells = N_cells
        self.move_features = move_features
        self.LinMix = LinMix
        self.meanbias = meanbias
        # Cell_NN = {'{}'.format(celln):nn.Sequential(nn.Linear(self.in_features, self.hidden_features),nn.Softplus(),nn.Linear(self.hidden_features, 1)) for celln in range(N_cells)}
        # self.visNN = nn.ModuleDict(Cell_NN)
        self.Cell_NN = nn.Sequential(nn.Linear(self.in_features, self.hidden_features,bias=True),nn.Softplus(),nn.Linear(self.hidden_features, N_cells,bias=True))
        self.activations = nn.ModuleDict({'SoftPlus':nn.Softplus(),
                                          'ReLU': nn.ReLU(),})
        # torch.nn.init.uniform_(self.Cell_NN[0].weight, a=-1e-6, b=1e-6)
        # torch.nn.init.constant_(self.Cell_NN[2].weight, 1)
        if self.meanbias is not None:
            torch.nn.init.constant_(self.Cell_NN[2].bias, meanbias)
        # Initialize Regularization parameters
        self.reg_alph = reg_alph
        if self.reg_alph != None:
            self.alpha = reg_alph*torch.ones(1).to(device)

        # Initialize Movement parameters
        self.reg_alphm = reg_alphm
        if self.move_features != None:
            if reg_alphm != None:
                self.alpha_m = reg_alphm*torch.ones(1).to(device)
            self.posNN = nn.ModuleDict({'Layer0': nn.Linear(move_features, N_cells)})
            torch.nn.init.uniform_(self.posNN['Layer0'].weight,a=-1e-6,b=1e-6)
            torch.nn.init.zeros_(self.posNN['Layer0'].bias)
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
            shift = Affine(angle=dxy[:,-1],translation=dxy[:,:2])
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
                output.append(self.visNN['{}'.format(celln)](inputs))
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
                move_out = torch.exp(move_out)
                output = output*move_out
        ret = self.activations['ReLU'](output)
        return ret
    
    def loss(self,Yhat, Y): 
        if self.LinMix:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
        else:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
            # loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0)  # Log-likelihood
        if self.move_features != None:
            if self.reg_alph != None:
                l1_reg0 = self.alpha0*(torch.linalg.norm(self.visNN['Layer0'].weight,axis=1,ord=1))
                l1_reg1 = self.alpha1*(torch.linalg.norm(self.visNN['Layer1'].weight,axis=1,ord=1))
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
                l1_reg0 = torch.stack([torch.linalg.vector_norm(NN_params,ord=1) for name, NN_params in self.visNN.named_parameters() if '0.weight' in name])
                l1_reg2 = torch.stack([torch.linalg.vector_norm(NN_params,ord=1) for name, NN_params in self.visNN.named_parameters() if '2.weight' in name])
                loss_vec = loss_vec + self.alpha*(l1_reg0 +l1_reg2)
        
        return loss_vec
