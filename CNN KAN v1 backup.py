import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
    #https://github.com/KindXiaoming/pykan/blob/master/kan/LBFGS.py
!pip install pykan
import torch
from functools import reduce
from torch.optim import Optimizer
import torch.nn as nn
from kan import *

__all__ = ['LBFGS']

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        #print(f_prev, f_new, g_new)
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1


    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old low becomes new high
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    #print(bracket)
    if len(bracket) == 1:
        t = bracket[0]
        f_new = bracket_f[0]
        g_new = bracket_g[0]
    else:
        t = bracket[low_pos]
        f_new = bracket_f[low_pos]
        g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals



class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 tolerance_ys=1e-32,
                 history_size=100,
                 line_search_fn=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            tolerance_ys=tolerance_ys,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        device = views[0].device
        return torch.cat(views, dim=0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad


    @torch.no_grad()
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """

        torch.manual_seed(0)

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        tolerance_ys = group['tolerance_ys']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > tolerance_ys:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)
                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss




class CNNKAN(nn.Module):
    def __init__(self, width1, grid1, k1, width2, grid2, k2):
        super(CNNKAN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        #self.kan1 = KANLinear(64 * 8 * 8, 256)   #in features, out features, grid, k
        #self.kan2 = KANLinear(256, 10)
        #self.kan1 = KAN(width=width1, grid=grid1, k=k1, device=device, sparse_init=False) #width=[3136, 256]
        #self.kan2 = KAN(width=width2, grid=grid2, k=k2, device=device, sparse_init=False) #width=[256, 10]
        self.kan1 = KANLinear(width1[0], width1[-1], grid1, k1)
        self.kan2 = KANLinear(width2[0], width2[-1], grid2, k2)


    def forward(self, x):
        print("-1-", x.shape)
        x = F.selu(self.conv1(x))
        print("-2-", x.shape)
        x = self.pool1(x)
        print("-3-", x.shape)
        x = F.selu(self.conv2(x))
        print("-4-", x.shape)
        x = self.pool2(x)
        print("-5-", x.shape)
        x = x.view(x.size(0), -1)
        print("-6-", x.shape)
        x = self.kan1(x)
        print("-7-", x.shape)
        x = self.kan2(x)
        print("-8-", x.shape)
        return x
    from google.colab import files
from google.colab import drive
drive.mount('/content/drive')

from sklearn import model_selection
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from tensorflow.keras.datasets import mnist
from datetime import datetime
from torchvision import datasets, transforms
import pickle
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

def string_datetime():
    return str(datetime.now()).replace(":", "").replace("-", "").replace(".", "")

PATIENCE_VALUE = 1
TOLERANCE_AMOUNT = 0.1
LOG_FILE_FILEPATH = '/content/drive/My Drive/text_logs/' + string_datetime() + '.txt'
#LOG_FILE_FILEPATH = './text_logs/' + string_datetime() + '.txt'
STEPS = 150
BATCH_SIZE = 1000
DATA_FILE_FILEPATH = '/content/drive/My Drive/data'
SAVE_FILES = False

KAN_CNN_MNIST_WIDTH_ORIGIN = 3136
KAN_CNN_MNIST_WIDTH_END = 10


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dtype = torch.get_default_dtype()
result_type = torch.long

with open(LOG_FILE_FILEPATH, 'w') as f:
    line = string_datetime() + '\n'
    f.write(line)
    f.close()

def write_to_file(line):
    line = line + '\n'
    with open(LOG_FILE_FILEPATH, "a") as f:
        f.write(line)
        f.close()

def log_csv(values):
    csv_file = LOG_FILE_FILEPATH[:-4] + "_csv.txt"
    line = ""
    for el in values:
      line = line + str(el) + ";"
    line = line[0:-1] + "\n"
    with open(csv_file, "a") as f:
        f.write(line)
        f.close()

def microseconds_since_2025(date_string):
    #Turns curdate into the amount of miliseconds since 01.04.2025
    #20250415 042707608044
    #yyyymmdd hhmmss
    date_format = "%Y%m%d %H%M%S%f"
    current_datetime = datetime.strptime(date_string, date_format)
    #reference datetime
    reference_datetime = datetime(2025, 4, 1)
    difference = (current_datetime - reference_datetime).total_seconds() * 1_000_000
    return int(difference)

def time_difference(start, end):
    return str(microseconds_since_2025(end) - microseconds_since_2025(start))

def count_kan_layer_parameters(in_dim, out_dim, KAN_degree, KAN_grid):
  return (in_dim * out_dim) * (KAN_grid + KAN_degree + 3) + out_dim

def count_mlp_layer_parameters(in_dim, out_dim):
  return (in_dim * out_dim) + out_dim

def make_mlp_min_layers(no_layers, layers_width):
  layers = []
  for i in range(no_layers):
    layers.append(layers_width)
  model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=layers, activation='relu', solver='lbfgs', max_iter=STEPS, tol=TOLERANCE_AMOUNT, early_stopping=True, validation_fraction=0.15, n_iter_no_change=PATIENCE_VALUE, batch_size=BATCH_SIZE)
  return model

def decide_min_mlp_layers_and_make(in_dim, out_dim, no_kan_parameters, how_many_layers_is_min=2):
  layer_width = int(in_dim/2)
  normal_layer_depth = how_many_layers_is_min
  current_no_params = count_mlp_layer_parameters(in_dim, layer_width) + count_mlp_layer_parameters(layer_width, out_dim) + normal_layer_depth * count_mlp_layer_parameters(layer_width, layer_width)

  while current_no_params < no_kan_parameters:
    layer_width += 1
    current_no_params = count_mlp_layer_parameters(in_dim, layer_width) + count_mlp_layer_parameters(layer_width, out_dim) + normal_layer_depth * count_mlp_layer_parameters(layer_width, layer_width)

  return make_mlp_min_layers(normal_layer_depth, layer_width-1), normal_layer_depth, layer_width-1, False

def create_matching_mlp_min_layers(KAN_width, KAN_degree, KAN_grid, how_many_layers_is_min=2):
  kan_sum = 0
  for i in range(len(KAN_width) - 1):
    kan_sum += count_kan_layer_parameters(KAN_width[i], KAN_width[i+1], KAN_degree, KAN_grid)

  return decide_min_mlp_layers_and_make(KAN_width[0], KAN_width[-1], kan_sum, how_many_layers_is_min)

def decide_min_plus_mlp_layers_and_make(in_dim, out_dim, no_kan_parameters, how_many_layers_is_min=2):
  layer_width = int(in_dim/2)
  normal_layer_depth = how_many_layers_is_min

  current_no_params = count_mlp_layer_parameters(in_dim, layer_width) + count_mlp_layer_parameters(layer_width,out_dim) + normal_layer_depth * count_mlp_layer_parameters(layer_width, layer_width)
  while current_no_params < no_kan_parameters:
    layer_width += 1
    current_no_params = count_mlp_layer_parameters(in_dim, layer_width) + count_mlp_layer_parameters(layer_width, out_dim) + normal_layer_depth * count_mlp_layer_parameters(layer_width, layer_width)

  return make_mlp_min_layers(normal_layer_depth, layer_width), normal_layer_depth, layer_width, False

def create_matching_mlp_min_plus_layers(KAN_width, KAN_degree, KAN_grid, how_many_layers_is_min=2):
  kan_sum = 0
  for i in range(len(KAN_width) - 1):
    kan_sum += count_kan_layer_parameters(KAN_width[i], KAN_width[i+1], KAN_degree, KAN_grid)

  return decide_min_plus_mlp_layers_and_make(KAN_width[0], KAN_width[-1], kan_sum, how_many_layers_is_min)

def count_kan_params(KAN_width1, KAN_degree1, KAN_grid1, KAN_width2, KAN_degree2, KAN_grid2):
  kan_sum = 0
  for i in range(len(KAN_width1) - 1):
    kan_sum += count_kan_layer_parameters(KAN_width1[i], KAN_width1[i+1], KAN_degree1, KAN_grid1)
  for i in range(len(KAN_width2) - 1):
    kan_sum += count_kan_layer_parameters(KAN_width2[i], KAN_width2[i+1], KAN_degree2, KAN_grid2)
  return kan_sum

def create_matching_mlp(plus, KAN_width1, KAN_degree1, KAN_grid1, KAN_width2, KAN_degree2, KAN_grid2):
  kan_sum = count_kan_params(KAN_width1, KAN_degree1, KAN_grid1, KAN_width2, KAN_degree2, KAN_grid2)
  return make_matching_mlp(plus, kan_sum, KAN_width1[0], KAN_width1[-1], KAN_width2[0], KAN_width2[-1])

def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_data):
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct/len(test_loader.dataset)


    line = "Eval_results: Average loss: " + str(test_loss) + " Accuracy: " + str(accuracy)
    write_to_file(line)

    return accuracy

def processKerasDatasetMNIST(random_state=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = datasets.MNIST(root=DATA_FILE_FILEPATH, train=True, download=True, transform=transform) #[0][1] -> int, 6000 2 1 28 28
    test_dataset = datasets.MNIST(root=DATA_FILE_FILEPATH, train=False, download=True, transform=transform)
    no_samples = len(train_dataset)
    len_of_val = int(0.2 * no_samples)

    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [no_samples - len_of_val, len_of_val]) #generator=torch.Generator().manual_seed(1)

    return train_subset, val_subset, test_dataset

def trainKAN(train_data, test_data, model, steps=1):
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    #optimizer = LBFGS(model.parameters(), history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    patience = PATIENCE_VALUE
    best_acc = 0
    best_model = model
    for i in range(steps):
        train(model, train_loader, optimizer)
        accuracy = evaluate(model, test_data)
        if accuracy < best_acc + TOLERANCE_AMOUNT:
            if patience == 0:
                return best_model, best_acc, i + 1
            else:
                patience -= 1
        else:
            patience = PATIENCE_VALUE
            best_acc = accuracy
            best_model = model

    return best_model, best_acc, steps

def trainMLP(dataset, model):
  model = model.fit(dataset['train_input'], dataset['train_label'])
  return model

skip = 5 + 30 + 21
# -----------------------------------
SAVE_FILES = False
dataset_name = "Mnist"
train_subset, val_subset, test_dataset = processKerasDatasetMNIST()

write_to_file(dataset_name)

log_csv(["dataset_name", "kan_grid 1", "kan_degree 1", "kan_lam 1", "kan_width 1", "kan_grid 2", "kan_degree 2", "kan_lam 2", "kan_width 2", "kan_time_difference", "kan_params", "kan_val_acc", "kan_test_acc", "kan_steps"])

line = "Skipping " + str(skip) + " lines"
write_to_file(line)



KAN_width_table = [256]
KAN_grid_table = [3, 5, 9]
KAN_degree_table = [2, 3, 4, 5]
KAN_lambda_table = [0]

#for i in range(int(in_dim * 0.75), (2*in_dim)+3, 8):
#    current_width = [in_dim, i, out_dim]
#    KAN_width_table.append(current_width)

for grid in KAN_grid_table:
    for degree in KAN_degree_table:
        for lam in KAN_lambda_table:
            for width in KAN_width_table:
              for i in range(5):
                if skip > 0:
                  skip -= 1
                  continue

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                kan_start = string_datetime()
                intro_line = "------------- start ---- grid: " + str(grid) + " deg: " + str(degree) + " lambda: " + str(lam) + " width: " + str(width) + " DT: " + kan_start
                write_to_file(intro_line)

                width1 = [KAN_CNN_MNIST_WIDTH_ORIGIN, width]
                width2 = [width, KAN_CNN_MNIST_WIDTH_END]

                model = CNNKAN(width1, grid, degree, width2, grid, degree).to(device)
                best_model, kan_val_acc, steps = trainKAN(train_subset, val_subset, model, STEPS)
                kan_test_acc = evaluate(best_model, test_dataset)

                first_train_line = "normal train: " + str(steps) + " steps; best val acc: " + str(kan_val_acc) + "; best TEST acc: " + str(kan_test_acc)
                write_to_file(first_train_line)

                kan_end = string_datetime()
                kan_params = count_kan_params(width1, degree, grid, width2, degree, grid)

                log_csv([dataset_name, str(grid), str(degree), str(lam), str(width1), str(grid), str(degree), str(lam), str(width2), time_difference(kan_start, kan_end), kan_params, str(kan_val_acc), str(kan_test_acc), steps])


print("I'm done accually!")


