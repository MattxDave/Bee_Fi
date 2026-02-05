# bee_policy.py - ATTENTION MODEL VERSION
import math

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(
        self,
        num_bees: int,
        action_dim: int = 3,
        hidden_dim=256,
        num_flowers: int = 12,
        retask_board_size: int = 0,
    ):
        super().__init__()
        self.num_bees = num_bees
        self.action_dim = action_dim
        self.num_flowers = int(num_flowers)
        self.hidden_dim = hidden_dim
        self.retask_board_size = int(retask_board_size)

        # --- Feature Encoders ---
        self.position_fc = nn.Sequential(nn.Linear(3, 64), nn.LayerNorm(64), nn.ReLU())
        self.status_fc = nn.Sequential(nn.Linear(2, 32), nn.LayerNorm(32), nn.ReLU())
        self.step_fc = nn.Sequential(nn.Linear(1, 32), nn.LayerNorm(32), nn.ReLU())
        self.cons_fc = nn.Sequential(nn.Linear(num_bees, 64), nn.LayerNorm(64), nn.ReLU())
        self.action_availability_fc = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.ReLU())

        # --- NEW: Attention-Based Flower Encoder ---
        # We will embed each flower's 12 features (with time window info) into a 128-dim vector
        self.flower_embed_dim = 128
        self.flowers_fc = nn.Sequential(
            nn.Linear(
                12, self.flower_embed_dim
            ),  # 12 features: 9 original + 3 time window features
            nn.LayerNorm(self.flower_embed_dim),
            nn.ReLU(),
        )

        # A 1-layer Transformer to process the set of flowers (faster)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.flower_embed_dim,
            nhead=4,  # 4 attention heads
            dim_feedforward=256,
            batch_first=True,  # Expect (B, N_flowers, embed_dim)
        )
        self.flower_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # --- End of New Section ---

        # Optional retask board encoder (if retask_board_size > 0)
        self.retask_board_fc = None
        retask_board_dim = 0
        if retask_board_size > 0:
            # Each retask slot: x, y, priority, reachable_flag, assigned_flag = 5 features
            retask_input_dim = 5 * retask_board_size
            self.retask_board_fc = nn.Sequential(
                nn.Linear(retask_input_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )
            retask_board_dim = 128

        # Calculate input dimension properly
        # The 256 from the old flower_fc is now self.flower_embed_dim
        trunk_input = 64 + 32 + self.flower_embed_dim + 32 + 64 + retask_board_dim + 32

        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Optional claim head (if retask_board_size > 0): outputs logits for M+1 slots
        self.claim_head = None
        if retask_board_size > 0:
            self.claim_head = nn.Linear(hidden_dim, retask_board_size + 1)  # +1 for no-claim

        # Better initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use a smaller gain for ReLU stability
            torch.nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, obs_dict, return_claims: bool = False):
        def ensure2d(x: torch.Tensor) -> torch.Tensor:
            x = x if x.dim() == 2 else x.view(1, -1)
            return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize inputs for stability
        position = ensure2d(obs_dict["position"]) / 30.0  # grid size normalization
        status = ensure2d(obs_dict["status"])

        # --- NEW: Flower Attention Logic ---
        flowers = ensure2d(obs_dict["flowers"])
        B = flowers.shape[0]

        # 1. Reshape to (Batch, NumFlowers, 12 Features)
        try:
            flowers_reshaped = flowers.view(B, self.num_flowers, 12)
        except RuntimeError as e:
            print(
                f"Policy ERROR: Flower input shape mismatch. Expected {self.num_flowers * 12} features, got {flowers.shape[1]}"
            )
            raise e

        # 2. Normalize features
        flowers_reshaped[:, :, 0] /= 30.0  # x coord (grid size)
        flowers_reshaped[:, :, 1] /= 30.0  # y coord (grid size)
        flowers_reshaped[:, :, 2] /= 10.0  # pollen (normalized by capacity)
        # features 3-7 are 0/1 flags (harvested, mine, busy, reachable, fits)
        # feature 8: dist_norm - already normalized by harvest_radius in env
        # feature 9: is_harvestable_now - 0/1 flag (time window check)
        # feature 10: is_hard_window - 0/1 flag
        # feature 11: time_to_window - already normalized by max_steps in env

        # 3. Embed each flower independently
        # Input: (B, N, 12) -> Output: (B, N, 128)
        flower_embed = self.flowers_fc(flowers_reshaped)

        # 4. Pass through Transformer
        # Input: (B, N, 128) -> Output: (B, N, 128)
        attended_flowers = self.flower_transformer(flower_embed)

        # 5. Pool the results. Mean pooling is simple and effective.
        # This aggregates all flower info into a single vector.
        # Input: (B, N, 128) -> Output: (B, 128)
        flw_out = attended_flowers.mean(dim=1)
        # --- End of New Flower Logic ---

        step_count = ensure2d(obs_dict["step_count"])
        consensus = ensure2d(obs_dict["consensus"])

        # Forward pass with normalized inputs
        pos_out = self.position_fc(position)
        sta_out = self.status_fc(status)
        # flw_out is now the (B, 128) vector from the transformer
        stp_out = self.step_fc(step_count)
        con_out = self.cons_fc(consensus)

        # Optional retask board encoding
        rtb_out = None
        if self.retask_board_fc is not None:
            retask_board = ensure2d(obs_dict.get("retask_board", None))
            if retask_board is not None and retask_board.numel() > 0:
                rtb_out = self.retask_board_fc(retask_board)
            else:
                # No retask board provided; use zero
                rtb_out = torch.zeros(B, 128, device=pos_out.device, dtype=pos_out.dtype)

        # Concatenate and process
        if rtb_out is not None:
            act_avail = (
                ensure2d(obs_dict.get("action_availability", None))
                if "action_availability" in obs_dict
                else None
            )
            if act_avail is not None:
                aa_out = self.action_availability_fc(act_avail)
            else:
                aa_out = torch.zeros(
                    pos_out.shape[0], 32, device=pos_out.device, dtype=pos_out.dtype
                )
            h = torch.cat([pos_out, sta_out, flw_out, stp_out, con_out, rtb_out, aa_out], dim=-1)
        else:
            act_avail = (
                ensure2d(obs_dict.get("action_availability", None))
                if "action_availability" in obs_dict
                else None
            )
            if act_avail is not None:
                aa_out = self.action_availability_fc(act_avail)
            else:
                aa_out = torch.zeros(
                    pos_out.shape[0], 32, device=pos_out.device, dtype=pos_out.dtype
                )
            h = torch.cat([pos_out, sta_out, flw_out, stp_out, con_out, aa_out], dim=-1)

        h = self.trunk(h)

        action_logits = self.policy_head(h)

        if return_claims and self.claim_head is not None:
            claim_logits = self.claim_head(h)
            return action_logits, claim_logits
        return action_logits


class CentralizedCritic(nn.Module):
    def __init__(
        self, global_state_size: int, num_bees: int, action_dim: int = 3, hidden_dim: int = 512
    ):
        super().__init__()
        self.global_state_size = int(global_state_size)
        self.num_bees = int(num_bees)
        self.action_dim = int(action_dim)
        # This part of the critic is not used in your PPO, but we leave it
        self.joint_action_dim = self.num_bees * self.action_dim
        self.hidden_dim = hidden_dim

        # Your critic only takes the global state, not joint action
        # This is fine for PPO. Let's fix the input dim.
        input_dim = self.global_state_size  # (was self.global_state_size + self.joint_action_dim)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.v_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, *args, **kwargs):
        state = None

        # Parse arguments
        if len(args) >= 1:
            state = args[0]
        if state is None and "state" in kwargs:
            state = kwargs["state"]

        if state is None:
            raise TypeError("CentralizedCritic.forward() missing required argument 'state'")

        # Ensure 2D and normalize
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Normalize state
        state_norm = state / 30.0  # grid size normalization

        # Your PPO logic does not pass joint_action, so we only use state
        x = torch.nan_to_num(state_norm, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.trunk(x)
        v = self.v_head(h)
        return v
