import torch
import torch.nn as nn
from kornia.geometry.transform import Affine
# torch.backends.cudnn.benchmark = True
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoissonGLM_VM_staticreg_shifter(nn.Module):
    def __init__(self, in_features, out_features, shift_in=3, shift_hidden=20, bias=True, reg_lam=None, reg_alph=None, reg_alphm=None, move_features=None, meanfr=None, init_sta=None, LinNetwork=False, device='cuda'):
        super(PoissonGLM_VM_staticreg_shifter, self).__init__()
        self.move_features = move_features
        self.LinNetwork = LinNetwork
        self.shift_in = shift_in
        self.shifter_nn = nn.Sequential(
            nn.Linear(self.shift_in,shift_hidden),
            nn.Softplus(),
            nn.Linear(shift_hidden, shift_in)
        )
        if self.move_features != None:
            if reg_lam != None:
                self.lam_m = reg_alph*torch.ones(out_features).to(device)
            if reg_alphm != None:
                self.alpha_m = reg_alphm*torch.ones(out_features).to(device)
            self.move_weights = nn.Parameter(torch.zeros(out_features,move_features), requires_grad=True)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if init_sta != None:
            self.weight = torch.nn.Parameter(init_sta, requires_grad=True)
            self.init_sta = True
        else:
            self.init_sta = False
            self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features),)
        self.reg_lam = reg_lam
        self.reg_alph = reg_alph
        if bias:
            if meanfr != None:
                self.bias = torch.nn.Parameter(meanfr,requires_grad=True)
                self.meanfr = True
            else:
                self.meanfr = None
                self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if self.reg_lam != None:
            self.lam = reg_lam*torch.ones(out_features).to(device)
        if self.reg_alph != None:
            self.alpha = reg_alph*torch.ones(out_features).to(device)

        self.reset_parameters()
        
    def reset_parameters(self):
        if self.init_sta == False:
            torch.nn.init.kaiming_uniform_(self.weight) #, a=np.sqrt(5)       
        if self.bias is not None:
            if self.meanfr == None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / torch.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, eye_input, move_input=None):
        batchsize, timesize, x, y = inputs.shape
        dxy = self.shifter_nn(eye_input)
        shift = Affine(angle=dxy[:,-1],translation=dxy[:,:2])
        inputs = shift(inputs)
        inputs = inputs.reshape(batchsize,-1).contiguous()
        x, y = inputs.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = inputs.matmul(self.weight.t())
        if move_input != None:
            output = output + move_input.matmul(self.move_weights.t())
        if self.bias is not None:
            output = output + self.bias
        if self.LinNetwork==True:
            ret = torch.relu(output)
        else:
            ret = torch.exp(output)
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def loss(self,Yhat, Y): 
        if self.LinNetwork:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
#             loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0) 
        else:
            loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0) 
        if self.move_features != None:
            if self.reg_alph != None:
                l1_regm = self.alpha_m*(torch.linalg.norm(self.weight[:,-self.move_features:],axis=1,ord=1))
                l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
            else: 
                l1_regm = 0
                l1_reg = 0
            loss_vec = loss_vec + l1_reg + l1_regm
        else:
            if self.reg_lam != None:
                if self.reg_alph != None:
                    l2_reg = self.lam*(torch.linalg.norm(self.weight,axis=1,ord=2))
                    l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
                    loss_vec = loss_vec + l2_reg + l1_reg
                else:
                    l2_reg = self.lam*(torch.linalg.norm(self.weight,axis=1,ord=2))
                    loss_vec = loss_vec + l2_reg
            else:
                if self.reg_alph != None:
                    l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
                    loss_vec = loss_vec + l1_reg
        
        return loss_vec


class PoissonGLM_VM_staticreg(nn.Module):
    def __init__(self, in_features, out_features, bias=True, reg_lam=None, reg_alph=None, reg_alphm=None, move_features=None, meanfr=None, init_sta=None, LinNetwork=False, device='cuda'):
        super(PoissonGLM_VM_staticreg, self).__init__()
        self.LinNetwork = LinNetwork
        self.move_features = move_features
        if self.move_features != None:
            if reg_lam != None:
                self.lam_m = reg_alph*torch.oness(out_features).to(device)
            if reg_alphm != None:
                self.alpha_m = reg_alphm*torch.ones(out_features).to(device)
            self.move_weights = nn.Parameter(torch.zeros(out_features,move_features), requires_grad=True)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if init_sta != None:
            self.weight = torch.nn.Parameter(init_sta, requires_grad=True)
            self.init_sta = True
        else:
            self.init_sta = False
            self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features),)
        self.reg_lam = reg_lam
        self.reg_alph = reg_alph
        if bias:
            if meanfr != None:
                self.bias = torch.nn.Parameter(meanfr,requires_grad=True)
                self.meanfr = True
            else:
                self.meanfr = None
                self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if self.reg_lam != None:
            self.lam = reg_lam*torch.ones(out_features).to(device)
        if self.reg_alph != None:
            self.alpha = reg_alph*torch.ones(out_features).to(device)
            
        if LinNetwork==True:
            self.lossfn = torch.nn.MSELoss()
        else: 
            self.lossfn = torch.nn.PoissonNLLLoss(log_input=True,reduction='mean')
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.init_sta == False:
            torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            if self.meanfr == None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / torch.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, inputs, move_input=None):
        x, y = inputs.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = inputs.matmul(self.weight.t())
        if move_input != None:
            output = output + move_input.matmul(self.move_weights.t())
        if self.bias is not None:
            output = output + self.bias
        if self.LinNetwork==True:
            ret = torch.relu(output)
#             ret = torch.log1p(torch.exp(output))
        else:
            ret = torch.exp(output)
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def loss(self,Yhat, Y): 
        if self.LinNetwork:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
#             loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0) 
        else:
            loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0) 
        if self.move_features != None:
            if self.reg_alph != None:
                l1_regm = self.alpha_m*(torch.linalg.norm(self.weight[:,-self.move_features:],axis=1,ord=1))
                l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
            else: 
                l1_regm = 0
                l1_reg = 0
            loss_vec = loss_vec + l1_reg + l1_regm
        else:
            if self.reg_lam != None:
                if self.reg_alph != None:
                    l2_reg = self.lam*(torch.linalg.norm(self.weight,axis=1,ord=2))
                    l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
                    loss_vec = loss_vec + l2_reg + l1_reg
                else:
                    l2_reg = self.lam*(torch.linalg.norm(self.weight,axis=1,ord=2))
                    loss_vec = loss_vec + l2_reg
            else:
                if self.reg_alph != None:
                    l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
                    loss_vec = loss_vec + l1_reg
        
        return loss_vec


class PoissonGLM_AddMult(nn.Module):
    def __init__(self, in_features, out_features, bias=True, reg_lam=None, reg_alph=None, reg_alphm=None, move_features=None, meanfr=None, init_sta=None, LinNetwork=False, device='cuda'):
        super(PoissonGLM_AddMult, self).__init__()
        self.move_features = move_features
        self.LinNetwork = LinNetwork

        if self.move_features != None:
            if reg_lam != None:
                self.lam_m = reg_alph*torch.ones(out_features).to(device)
            if reg_alphm != None:
                self.alpha_m = reg_alphm*torch.ones(out_features).to(device)
            self.move_weights = nn.Parameter(1e-6*torch.ones(out_features,move_features), requires_grad=True)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if init_sta != None:
            self.weight = torch.nn.Parameter(init_sta, requires_grad=True)
            self.init_sta = True
        else:
            self.init_sta = False
            self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features),)
        self.reg_lam = reg_lam
        self.reg_alph = reg_alph
        if bias:
            if meanfr != None:
                self.bias = torch.nn.Parameter(meanfr,requires_grad=True)
                self.meanfr = True
            else:
                self.meanfr = None
                self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if self.reg_lam != None:
            self.lam = reg_lam*torch.ones(out_features).to(device)
        if self.reg_alph != None:
            self.alpha = reg_alph*torch.ones(out_features).to(device)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.init_sta == False:
            torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            if self.meanfr == None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / torch.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, move_input=None):
        x, y = inputs.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = inputs.matmul(self.weight.T)
        if self.bias is not None:
            output = output + self.bias
        if self.LinNetwork==True:
            if move_input != None:
                output = output + move_input.matmul(self.move_weights.T)
            ret = torch.relu(output)
        else:
            if move_input != None:
                output = output*move_input.matmul(self.move_weights.T)
            ret = torch.relu(output)
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def loss(self,Yhat, Y): 
        if self.LinNetwork:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
#             loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0) 
        else:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
            # loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0) 
        if self.move_features != None:
            if self.reg_alph != None:
                l1_regm = self.alpha_m*(torch.linalg.norm(self.weight[:,-self.move_features:],axis=1,ord=1))
                l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
            else: 
                l1_regm = 0
                l1_reg = 0
            loss_vec = loss_vec + l1_reg + l1_regm
        else:
            if self.reg_lam != None:
                if self.reg_alph != None:
                    l2_reg = self.lam*(torch.linalg.norm(self.weight,axis=1,ord=2))
                    l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
                    loss_vec = loss_vec + l2_reg + l1_reg
                else:
                    l2_reg = self.lam*(torch.linalg.norm(self.weight,axis=1,ord=2))
                    loss_vec = loss_vec + l2_reg
            else:
                if self.reg_alph != None:
                    l1_reg = self.alpha*(torch.linalg.norm(self.weight,axis=1,ord=1))
                    loss_vec = loss_vec + l1_reg
        
        return loss_vec
