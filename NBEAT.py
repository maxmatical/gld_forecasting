import torch
import torch.nn as nn
import torch.nn.functional as F  
from fastai import *
from fastai.tabular import *

# activation functions
act_fn = nn.ReLU(inplace=True)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))

mish = Mish()

# block within NBEAT
class block(Module):
    def __init__(self, ni, nh, theta_dim, n_out, bn:bool = True, ps:float=0., actn:Optional[nn.Module]=None):
        """
        ni = backcast length
        """
        super().__init__()
        layers =[*bn_drop_lin(ni,nh,bn,ps,actn),
                  *bn_drop_lin(nh,nh,bn,ps,actn),
                  *bn_drop_lin(nh,nh,bn,ps,actn),
                  *bn_drop_lin(nh,nh,bn,ps,actn)]
        self.ff_block = nn.Sequential(*layers)

        fwd_layers = [*bn_drop_lin(nh,theta_dim,bn,ps,actn), 
                      *bn_drop_lin(theta_dim,n_out,bn,ps, actn = None)]# no act fn on fwd and bwd forecast
        bwd_layers = [*bn_drop_lin(nh,theta_dim,bn,ps,actn),
                      *bn_drop_lin(theta_dim,ni,bn,ps, actn = None)]

        self.fwd = nn.Sequential(*fwd_layers) 
        self.bwd = nn.Sequential(*bwd_layers)

    def forward(self, x):
        x1 = self.ff_block(x)
        x_fwd = self.fwd(x1)
        x_bwd = self.bwd(x1)

        return(x-x_bwd, x_fwd)

class nbeats(Module):
    def __init__(self, nh, theta_dim, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False, n_blocks=6, act_fn=act_fn):
        super().__init__()
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        self.ni  = self.n_emb + self.n_cont
        self.n_out = out_sz
        self.nh = nh
        self.theta_dim = theta_dim
        self.n_blocks = n_blocks
        self.ps = ps
        self.bn = use_bn
        self.act_fn = act_fn

        if self.ps is None:
            self.ps = [0.]*self.n_blocks
        elif type(self.ps) == float:
            self.ps = [self.ps]*self.n_blocks
        assert len(self.ps) == self.n_blocks, "size mismatch"

        self.stack = nn.ModuleList()
        for i in range(self.n_blocks):
            self.stack.append(gen_block(self.ni, self.nh, self.theta_dim, self.n_out, bn=self.bn, ps=self.ps[i], actn=self.act_fn))


    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont

        out = 0 
        for i, b in enumerate(self.stack):
            x, x_fwd = b(x)
            out += x_fwd

        if self.y_range is not None: # squeezing to y_range
            out = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(out) + self.y_range[0]

        return out

def forecast_learner(data:DataBunch, nh, theta_dim, layers:Collection[int], emb_szs:Dict[str,int]=None, metrics=None,
        ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, n_blocks = 6, act_fn=act_fn, **learn_kwargs):
    "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = nbeats(nh, theta_dim, emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, ps=ps, emb_drop=emb_drop,
                    y_range=y_range, use_bn=use_bn, n_blocks = n_blocks, act_fn = act_fn)
    return Learner(data, model, metrics=metrics, **learn_kwargs)





