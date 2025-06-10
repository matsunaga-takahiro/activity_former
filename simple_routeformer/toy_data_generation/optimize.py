import numpy as np
from scipy.optimize import minimize
import torch
import torch.optim as optim

from utils.logger import logger
torch.autograd.set_detect_anomaly(True)

class GPUOptimizer:
    def __init__(self, network, trip, recursive, bounds, xinit, demands):
        logger.info('Optimizer init')
        self.network = network
        self.trip = trip
        self.recursive = recursive
        self.bounds = bounds
        self.x_opt = None
        self.LL0 = None
        self.LL_res = None
        self.rho2 = None
        self.rho2_adj = None
        self.tval_res = None
        self.device = self.recursive.device
        self.x = torch.nn.Parameter(torch.tensor(xinit, dtype=torch.float32, device=self.device))
        self.xinit = xinit
        self.demands = demands

        # Extract bounds as tensors
        self.min_bounds = torch.tensor([b[0] for b in self.bounds], dtype=torch.float32, device=self.device)
        self.max_bounds = torch.tensor([b[1] for b in self.bounds], dtype=torch.float32, device=self.device)

    def hessian(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True, device=self.device)

        def loss_fn(x):
            return -self._loglikelihood(x)

        hess = torch.autograd.functional.hessian(loss_fn, x_tensor)
        return hess.detach().cpu().numpy()

    def tval(self, x):
        hess = self.hessian(x)
        var = np.diag(np.linalg.inv(hess))
        t_values = x / np.sqrt(var)
        return t_values

    def optimize(self):
        logger.info('Optimizer optimize start')

        # Set up optimizer
        optimizer = optim.LBFGS([self.x], max_iter=10)

        def closure():
            optimizer.zero_grad()
            loss = -self._loglikelihood(self.x)
            loss.backward()
            if self.x.grad is None:
                raise ValueError("Gradient not calculated for x")
            logger.info(f"x.grad: {self.x.grad}")
            return loss

        optimizer.step(closure)

        # Clamp parameters to bounds
        with torch.no_grad():
            self.x = torch.clamp(self.x, self.min_bounds, self.max_bounds).detach()

        x_opt = self.x.detach().cpu().numpy()

        # t-values and LL calculations
        x0 = x_opt
        tval_res = self.tval(x0)
        LL0 = self._loglikelihood(torch.tensor(self.xinit, dtype=torch.float32, device=self.device)).item()
        LL_res = self._loglikelihood(self.x).item()

        rho2 = (LL0 - LL_res) / LL0
        rho2_adj = (LL0 - (LL_res - len(x0))) / LL0

        self.x_opt = x_opt
        self.LL0 = LL0
        self.LL_res = LL_res
        self.rho2 = rho2
        self.rho2_adj = rho2_adj
        self.tval_res = tval_res

        return x_opt, LL0, LL_res, rho2, rho2_adj, tval_res

    def _loglikelihood(self, x):
        """
        対数尤度関数
        """
        LLsum = torch.tensor(0.0, device=self.device)
        Pall = self.recursive.newPall(x)

        od_indices = self.trip['od_pair'].values  # OD のインデックス
        df_tensor = torch.tensor(
            self.trip[[str(t) for t in range(self.recursive.T + 1)]].values,
            device=self.device,
            dtype=torch.int64
        )
        for od in range(len(self.demands)):
            dfi_tensor = df_tensor[od_indices == od]
            for t in range(self.recursive.T):
                # k と a を抽出
                k = dfi_tensor[:, t]  # 時刻 t の値
                a = dfi_tensor[:, t + 1]  # 時刻 t+1 の値
                pka = Pall[od, t, k, a]
                pka = torch.where(pka <= 0, torch.tensor(1e-10, device=self.device), pka)
                #if pka > 1:
                    #logger.info(f'pka={pka}, od={od}, k={k}, a={a}')
                LLsum += torch.sum(torch.log(pka))

        logger.info(f'x={x}, LL={LLsum}')
        return LLsum