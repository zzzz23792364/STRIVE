import torch
from torch import nn

from losses.adv_gen_nusc import MotionPriorLoss, VehCollLoss, EnvCollLoss, interp_traj


class RLReward(nn.Module):
    def __init__(self, loss_weights, veh_att, mapixes, map_env, ptr):
        super().__init__()
        self.loss_weights = loss_weights
        self.motion_prior_loss = MotionPriorLoss()
        self.ptr = ptr
        self.graph_sizes = ptr[1:] - ptr[:-1]
        self.ego_mask = torch.zeros((veh_att.size(0)), dtype=torch.bool)
        self.ego_mask[ptr[:-1]] = True
        self.veh_coll_loss = VehCollLoss(veh_att, ptr=ptr)
        self.env_coll_loss = EnvCollLoss(
            veh_att[~self.ego_mask], mapixes[~self.ego_mask], map_env
        )

    def forward(self, future_pred, tgt_traj, z, prior_out):
        NA = future_pred.size(0)
        B = tgt_traj.size(0)
        loss = 0.0
        info = {}

        future_pred_interp = interp_traj(future_pred, scale_factor=3)
        NA_mask = ~self.ego_mask
        NA_count = NA_mask.sum().item()

        if (
            "adv_crash" in self.loss_weights
            and self.loss_weights["adv_crash"] > 0
            and NA_count > 0
        ):
            attacker_pred = future_pred[NA_mask][:, :, :2]
            tgt_pred = tgt_traj[:, :, :2]
            tgt_expanded = torch.cat(
                [
                    tgt_pred[b : b + 1].expand(self.graph_sizes[b] - 1, -1, -1)
                    for b in range(B)
                ],
                dim=0,
            )
            dist_traj = torch.norm(attacker_pred - tgt_expanded, dim=-1)
            min_dist = dist_traj.min(dim=-1)[0].min(dim=-1)[0].mean()
            loss = loss + self.loss_weights["adv_crash"] * min_dist
            info["crash_dist"] = min_dist.item()

        if (
            "motion_prior" in self.loss_weights
            and self.loss_weights["motion_prior"] > 0
        ):
            prior_loss = self.motion_prior_loss(z, prior_out).mean()
            loss = loss + self.loss_weights["motion_prior"] * prior_loss
            info["prior_nll"] = prior_loss.item()

        if (
            "coll_veh" in self.loss_weights
            and self.loss_weights["coll_veh"] > 0
        ):
            veh_loss = self.veh_coll_loss(future_pred_interp).mean()
            loss = loss + self.loss_weights["coll_veh"] * veh_loss
            info["veh_coll"] = veh_loss.item()

        if (
            "coll_env" in self.loss_weights
            and self.loss_weights["coll_env"] > 0
            and NA_count > 0
        ):
            env_loss = self.env_coll_loss(
                future_pred_interp[NA_mask]
            ).mean()
            loss = loss + self.loss_weights["coll_env"] * env_loss
            info["env_coll"] = env_loss.item()

        reward = -loss
        info["reward"] = reward.item()
        info["loss"] = loss.item()
        return reward, info
