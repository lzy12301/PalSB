import torch
from tqdm import tqdm


def get_sampling_func(net, ema, num_scales, nfe, diffusion, log_count=10, device='cpu', verbose=True, ot_ode=False, es=True):
    return lambda x, cond, cal_log_prob=False: sampling_func(x, cond, net, ema, num_scales, nfe, diffusion, log_count, device=device, verbose=verbose, cal_log_prob=cal_log_prob, ot_ode=ot_ode, es=es)


def get_controlled_sampling_func(net, ema, num_scales, nfe, diffusion, log_count=10, device='cpu', **kwargs):
    return lambda x, cond: controlled_sampling_func(x, cond, net, ema, num_scales, nfe, diffusion, log_count, device=device, **kwargs)


def sampling_func(x1, cond, net, ema, num_scales, nfe, diffusion, log_count=10, device='cpu', cal_log_prob=False, verbose=True, ot_ode=False, es=True):
    nfe = nfe or num_scales-1
    assert 0 < nfe < num_scales == len(diffusion.betas)
    if not es:
        steps = space_indices(num_scales, nfe+1)    # range(num_scales)[nfe:], space_indices(num_scales, nfe+1), range(num_scales)[-nfe-1:]
    else:
        steps = range(num_scales)[-nfe-1:]

    # create log steps
    log_count = min(len(steps)-1, log_count)
    log_steps = [steps[i] for i in space_indices(len(steps)-1, log_count)]
    # assert log_steps[0] == 0
    # print(f"[DDPM Sampling] steps={num_scales}, {nfe=}, {log_steps=}!")

    x1 = x1.to(device)
    if cond is not None:
        cond = cond.to(device)
    xt = x1.detach().to(device)
    pred_x0s = []
    xs = []
    log_probs = []
    if ema is not None:
        with ema.average_parameters():
            net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=device, dtype=torch.long)
                return net(xt, cond, step,)

            steps = steps[::-1]
            pair_steps = zip(steps[1:], steps[:-1])
            pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
            for prev_step, step in pair_steps:
                pred_x0 = pred_x0_fn(xt, prev_step)
                if not cal_log_prob:
                    xt = diffusion.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
                    # xt = diffusion.q_sample(prev_step, pred_x0, x1, ot_ode=ot_ode)
                else:
                    xt, log_prob = diffusion.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, cal_log_prob=cal_log_prob)
                # xt = diffusion.q_sample(step, pred_x0, x1, ot_ode=False)
                if prev_step in log_steps:
                    pred_x0s.append(pred_x0.detach().cpu())
                    xs.append(xt.detach().cpu())
                    if cal_log_prob:
                        log_probs.append(log_prob.detach().cpu())
    else:
        net.eval()

        def pred_x0_fn(xt, step):
            step = torch.full((xt.shape[0],), step, device=device, dtype=torch.long)
            return net(xt, cond, step,)

        steps = steps[::-1]
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            pred_x0 = pred_x0_fn(xt, prev_step)
            if not cal_log_prob:
                xt = diffusion.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            else:
                xt, log_prob = diffusion.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, cal_log_prob=cal_log_prob)
            # xt = diffusion.q_sample(step, pred_x0, x1, ot_ode=False)
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
                if cal_log_prob:
                    log_probs.append(log_prob.detach().cpu())
    return (torch.stack(xs, dim=1), torch.stack(pred_x0s, dim=1)) if not cal_log_prob else (torch.stack(xs, dim=1), torch.stack(pred_x0s, dim=1), torch.stack(log_probs, dim=1))


def controlled_sampling_func(x1, cond, net, ema, num_scales, nfe, diffusion, log_count=10, device='cpu', **kwargs):
    nfe = nfe or num_scales-1
    assert 0 < nfe < num_scales == len(diffusion.betas)
    steps = range(num_scales)[-nfe-1:]    # range(num_scales)[nfe:], space_indices(num_scales, nfe+1)

    # create log steps
    log_count = min(len(steps)-1, log_count)
    log_steps = [steps[i] for i in space_indices(len(steps)-1, log_count)]
    # assert log_steps[0] == 0
    print(f"[DDPM Sampling] steps={num_scales}, {nfe=}, {log_steps=}!")

    x1 = x1.to(device)
    if cond is not None:
        cond = cond.to(device)
    xt = x1.detach().to(device)
    pred_x0s = []
    xs = []
    is_control = False
    if 'net_control' in kwargs.keys():
        if kwargs['net_control'] is not None:
            is_control = True
            net_control = kwargs['net_control']
            control_func = kwargs['control_func']
            scalar_inv = kwargs['scalar_inv']
    with ema.average_parameters():
        net.eval()
        if is_control:
            net_control.eval()

        def pred_x0_fn(xt, step):
            step = torch.full((xt.shape[0],), step, device=device, dtype=torch.long)
            if is_control:
                hint = control_func(scalar_inv(xt))
                control = net_control(xt, x1, step, hint)
                return net(xt, cond, step, control=control, only_mid_control=False)
            else:
                return net(xt, cond, step,)

        steps = steps[::-1]
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1)
        for prev_step, step in pair_steps:
            pred_x0 = pred_x0_fn(xt, prev_step)
            xt = diffusion.p_posterior(prev_step, step, xt, pred_x0, ot_ode=False)
            # xt = diffusion.q_sample(step, pred_x0, x1, ot_ode=False)
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
    return torch.stack(xs, dim=1), torch.stack(pred_x0s, dim=1)


def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps
