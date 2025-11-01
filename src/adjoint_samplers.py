import torch
from typing import Union, List, Callable
import tqdm


def get_conditional_denoiser(
    net,
    *,
    second_of_day,
    day_of_year,
    labels,
    condition,
):
    def D(x_hat, t_hat, c=None, doy=None, sod=None):
        if c is None:
            c = condition
        if doy is None:
            doy = day_of_year
        if sod is None:
            sod = second_of_day

        net.eval()
        return net(
            x_hat,
            t_hat,
            labels,
            condition=c,
            second_of_day=sod,
            day_of_year=doy,
        ).out
    
    D.round_sigma = net.round_sigma
    D.sigma_max = net.sigma_max
    D.sigma_min = net.sigma_min
    return D


def edm_sampler_with_jvp(
    net,
    latent: torch.Tensor,
    condition: torch.Tensor,
    day_of_year: torch.Tensor,
    delta_sst_conditioning: torch.Tensor,
    delta_doy: torch.Tensor,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
):
    # patch_attention_op()
    assert delta_sst_conditioning.shape == condition.shape, (
        f"Expected SST tangent vectors shape {condition.shape}, got {delta_sst_conditioning.shape}."
    )
    assert delta_doy.shape == day_of_year.shape, (
        f"Expected DOY tangent vectors shape {day_of_year.shape}, got {delta_doy.shape}."
    )

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latent.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = net.round_sigma(t_steps)

    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = latent.to(torch.float64)
    w_next = torch.zeros_like(x_next)
    for i, (t_cur, t_next) in tqdm.tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps, desc="Sampling Steps"):
        x_cur = x_next
        adjoint_cur = w_next

        t_hat = net.round_sigma(t_cur)

        def velocity_constructor(t):
            # This function constructs the velocity function for the JVP.
            def velocity_from_denoised(x_hat, condition, doy):
                # Compute the denoised output and the velocity.
                denoised = net(x_hat, t, c=condition, doy=doy).to(torch.float64)
                d_cur = (x_hat - denoised) / t
                return d_cur
            return velocity_from_denoised

        # Compute JVP.
        u, d_t_w_next = torch.func.jvp(velocity_constructor(t_hat), (x_cur, condition, day_of_year), (adjoint_cur, delta_sst_conditioning, delta_doy))

        # Euler step.
        x_next = x_cur + (t_next - t_hat) * u
        w_next = adjoint_cur + (t_next - t_hat) * d_t_w_next

        # Apply 2nd order correction.
        if i < num_steps - 1:
            u_prime, d_t_w_next_prime = torch.func.jvp(velocity_constructor(t_next), (x_next, condition, day_of_year), (w_next, delta_sst_conditioning, delta_doy))
            x_next = x_cur + (t_next - t_hat) * (0.5 * u + 0.5 * u_prime)
            w_next = adjoint_cur + (t_next - t_hat) * (0.5 * d_t_w_next + 0.5 * d_t_w_next_prime)

    return x_next, w_next


def edm_reverse_sampler_with_conditioning_gradients(
    net,
    clean_image: torch.Tensor,
    condition: torch.Tensor,
    day_of_year: torch.Tensor,
    second_of_day: torch.Tensor,
    aggregation_fns: Union[List[Callable], Callable],
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Handle both single callable and list of callables
    if callable(aggregation_fns):
        aggregation_fns = [aggregation_fns]

    with torch.set_grad_enabled(True):
        # Ensure clean_image and conditioning have been detached and require gradients.
        clean_image = clean_image.detach().requires_grad_(True)
        condition = condition.detach().requires_grad_(True)

        # Compute q values and gradients for each aggregation function
        qs = []
        grad_qs = []
        # TODO vmap functions or batched backward.
        for aggregation_fn in aggregation_fns:
            q = aggregation_fn(clean_image)
            assert q.shape == (clean_image.shape[0], clean_image.shape[2]), f"Expected q shape {(clean_image.shape[0], clean_image.shape[2])}, got {q.shape}."
            # Get the gradient of the quantity of interest with respect to the clean_image.
            grad_q = torch.autograd.grad(
                outputs=q.sum(),    # A scalar output is needed, but it's the same as taking autograd on each of the batch coordinates. Only works if batches haven't been mixed by the aggregation function.
                inputs=clean_image,
            )[0]  # shape (B, C, t, hpx_idx)
            qs.append(q)
            grad_qs.append(grad_q)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=clean_image.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = net.round_sigma(t_steps)
    t_steps = torch.flip(t_steps, [0])

    # Main sampling loop.
    x_next = clean_image.to(torch.float64)
    adjoint_nexts = [
        grad_q.to(torch.float64)
        for grad_q in grad_qs
    ]
    adjoint_nexts = torch.stack(adjoint_nexts, dim=0)
    w_nexts = torch.stack([torch.zeros_like(condition) for _ in aggregation_fns], dim=0)
    v_nexts = torch.stack([torch.zeros_like(day_of_year) for _ in aggregation_fns], dim=0)
    h_nexts = torch.stack([torch.zeros_like(second_of_day) for _ in aggregation_fns], dim=0)

    progress_bar = tqdm.tqdm(total=num_steps, desc="Reverse Sampling Steps")
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        adjoint_curs = adjoint_nexts
        w_curs = w_nexts
        v_curs = v_nexts
        h_curs = h_nexts

        t_hat = net.round_sigma(t_cur)
        x_hat = x_cur

        def velocity_from_denoised(x_hat, t_hat, condition, day_of_year, second_of_day) -> torch.Tensor:
            # Compute the denoised output and the velocity.
            denoised = net(x_hat, t_hat, c=condition, doy=day_of_year, sod=second_of_day).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            return d_cur
        @torch.enable_grad()
        def get_grad_velocities(x_hat, t_hat, condition, day_of_year, second_of_day, adjoint_curs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            x_hat = x_hat.detach().requires_grad_(True)
            t_hat = t_hat.detach().requires_grad_(True)
            day_of_year = day_of_year.detach().requires_grad_(True)
            second_of_day = second_of_day.detach().requires_grad_(True)
            condition = condition.detach().requires_grad_(True)
            adjoint_curs = adjoint_curs.detach().requires_grad_(True)

            # Compute velocity once
            u = velocity_from_denoised(x_hat, t_hat, condition, day_of_year, second_of_day)

            # Pre-allocate stacked tensors for gradients
            dadt_stack = torch.zeros_like(adjoint_curs)
            dwdt_stack = torch.zeros_like(w_nexts)
            aT_dqdDoY_stack = torch.zeros((adjoint_curs.shape[0], adjoint_curs.shape[1], 1), device=adjoint_curs.device, dtype=adjoint_curs.dtype)  # shape (num_q, batch_size, 1), same as DoY
            aT_dqdToD_stack = torch.zeros((adjoint_curs.shape[0], adjoint_curs.shape[1], 1), device=adjoint_curs.device, dtype=adjoint_curs.dtype)  # shape (num_q, batch_size, 1), same as ToD
            
            # Avoid expanding all computational graphs in memory, do one q at a time.
            for idx in range(len(adjoint_curs)):
                # VJP not supported with checkpointing. Use autograd.grad.
                grads = torch.autograd.grad(
                    outputs=u,
                    inputs=(x_hat, condition, t_hat, day_of_year, second_of_day),
                    grad_outputs=-adjoint_curs[idx],
                    retain_graph=True if idx < len(adjoint_curs) - 1 else False,
                )

                dadt, dwdt, _, aT_dqdDoY, aT_dqdToD = grads
                dadt_stack[idx] = dadt
                dwdt_stack[idx] = dwdt
                aT_dqdDoY_stack[idx] = aT_dqdDoY
                aT_dqdToD_stack[idx] = aT_dqdToD

            return u.detach(), dadt_stack.detach(), dwdt_stack.detach(), aT_dqdDoY_stack.detach(), aT_dqdToD_stack.detach()

        u, dadt_stack, dwdt_stack, aT_dqdDoY_stack, aT_dqdToD_stack = get_grad_velocities(x_hat, t_hat, condition, day_of_year, second_of_day, adjoint_curs)
        x_next = x_hat + (t_next - t_hat) * u
        adjoint_nexts = adjoint_curs + (t_next - t_hat) * dadt_stack
        w_nexts = w_curs + (t_next - t_hat) * dwdt_stack
        v_nexts = v_curs + (t_next - t_hat) * aT_dqdDoY_stack
        h_nexts = h_curs + (t_next - t_hat) * aT_dqdToD_stack


        # Apply 2nd order correction.
        if i < num_steps - 1:
            u_prime, dadt_stack_prime, dwdt_stack_prime, aT_dqdDoY_stack_prime, aT_dqdToD_stack_prime = get_grad_velocities(x_next, t_next, condition, day_of_year, second_of_day, adjoint_nexts)
            x_next = x_hat + (t_next - t_hat) * (0.5 * u + 0.5 * u_prime)
            adjoint_nexts = adjoint_curs + (t_next - t_hat) * (0.5 * dadt_stack + 0.5 * dadt_stack_prime)
            w_nexts = w_curs + (t_next - t_hat) * (0.5 * dwdt_stack + 0.5 * dwdt_stack_prime)
            v_nexts = v_curs + (t_next - t_hat) * (0.5 * aT_dqdDoY_stack + 0.5 * aT_dqdDoY_stack_prime)
            h_nexts = h_curs + (t_next - t_hat) * (0.5 * aT_dqdToD_stack + 0.5 * aT_dqdToD_stack_prime)

        progress_bar.update(1)

    # Concatenate all w results along the channel dimension (dim 1)
    concatenated_ws = torch.cat([w_nexts[i] for i in range(w_nexts.shape[0])], dim=1)
    concatenated_vs = torch.cat([v_nexts[i] for i in range(v_nexts.shape[0])], dim=1).unsqueeze(-1) # add time dimension, [num_q, batch, 1]
    concatenated_hs = torch.cat([h_nexts[i] for i in range(h_nexts.shape[0])], dim=1).unsqueeze(-1) # add time dimension, [num_q, batch, 1]

    return x_next, qs, concatenated_ws, concatenated_vs, concatenated_hs