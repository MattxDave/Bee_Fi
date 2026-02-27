# hrl_policy.py - Hierarchical RL Policies
"""
Hierarchical RL for Bee Foraging:
- Manager (high-level): Assigns goals to each bee every N steps
- Worker (low-level): Executes actions to achieve assigned goals

Goals:
  0 = IDLE (wait for better opportunity)
  1 = HARVEST_TARGET (go harvest assigned flower)
  2 = GROOM (offload pollen or recharge)
  3 = ASSIST (help another bee's task from retask board)
"""

import math
import torch
import torch.nn as nn


class ManagerPolicy(nn.Module):
    """
    High-level policy that assigns goals to bees.
    
    Operates at a slower timescale (every manager_interval steps).
    Observes global state and outputs a goal for each bee.
    """
    
    # Goal definitions
    GOAL_IDLE = 0
    GOAL_HARVEST = 1
    GOAL_GROOM = 2
    GOAL_ASSIST = 3
    NUM_GOALS = 4
    
    def __init__(
        self,
        num_bees: int,
        num_flowers: int = 12,
        hidden_dim: int = 256,
        manager_interval: int = 10,
    ):
        super().__init__()
        self.num_bees = num_bees
        self.num_flowers = num_flowers
        self.hidden_dim = hidden_dim
        self.manager_interval = manager_interval
        
        # --- Global State Encoders ---
        # Encode all bee states together
        bee_input_dim = 3 + 2 + 1 + 1  # position(3) + status(2) + battery_ratio + load_ratio
        self.bee_encoder = nn.Sequential(
            nn.Linear(bee_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Encode all flower states with attention
        flower_input_dim = 12  # Same as worker observation
        self.flower_embed_dim = 64
        self.flower_encoder = nn.Sequential(
            nn.Linear(flower_input_dim, self.flower_embed_dim),
            nn.LayerNorm(self.flower_embed_dim),
            nn.ReLU(),
        )
        
        # Attention over flowers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.flower_embed_dim,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        self.flower_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Global context: step count, overall progress
        self.context_encoder = nn.Sequential(
            nn.Linear(3, 32),  # step_norm, flowers_harvested_ratio, active_bees_ratio
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        # Combine all encodings
        # Per-bee: 64, Flowers: 64, Context: 32
        combined_dim = 64 + self.flower_embed_dim + 32
        
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Output: goal logits for each bee
        # We output per-bee goals, so we need bee-specific heads
        self.goal_head = nn.Linear(hidden_dim, self.NUM_GOALS)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, global_obs: dict) -> torch.Tensor:
        """
        Args:
            global_obs: dict with:
                - bee_states: (B, num_bees, bee_input_dim) - all bee states
                - flowers: (B, num_flowers, 12) - all flower features
                - context: (B, 3) - global context [step_norm, harvest_ratio, active_ratio]
        
        Returns:
            goal_logits: (B, num_bees, NUM_GOALS) - goal logits per bee
        """
        bee_states = global_obs["bee_states"]  # (B, num_bees, 8)
        flowers = global_obs["flowers"]  # (B, num_flowers, 12)
        context = global_obs["context"]  # (B, 3)
        
        B = bee_states.shape[0]
        
        # Encode bees
        bee_embed = self.bee_encoder(bee_states)  # (B, num_bees, 64)
        
        # Encode flowers with attention
        flower_embed = self.flower_encoder(flowers)  # (B, num_flowers, 64)
        flower_attended = self.flower_transformer(flower_embed)  # (B, num_flowers, 64)
        flower_pooled = flower_attended.mean(dim=1)  # (B, 64)
        
        # Encode context
        context_embed = self.context_encoder(context)  # (B, 32)
        
        # Broadcast flower and context to each bee
        flower_broadcast = flower_pooled.unsqueeze(1).expand(-1, self.num_bees, -1)  # (B, num_bees, 64)
        context_broadcast = context_embed.unsqueeze(1).expand(-1, self.num_bees, -1)  # (B, num_bees, 32)
        
        # Combine per-bee
        combined = torch.cat([bee_embed, flower_broadcast, context_broadcast], dim=-1)  # (B, num_bees, 160)
        
        # Process through trunk (per-bee)
        combined_flat = combined.view(B * self.num_bees, -1)
        h = self.trunk(combined_flat)
        goal_logits = self.goal_head(h)
        
        # Reshape to (B, num_bees, NUM_GOALS)
        goal_logits = goal_logits.view(B, self.num_bees, self.NUM_GOALS)
        
        return goal_logits
    
    def get_goals(self, global_obs: dict, stochastic: bool = True) -> tuple:
        """
        Get goals for all bees.
        
        Returns:
            goals: (B, num_bees) - selected goal indices
            log_probs: (B, num_bees) - log probabilities of selected goals
        """
        goal_logits = self.forward(global_obs)  # (B, num_bees, NUM_GOALS)
        
        if stochastic:
            dist = torch.distributions.Categorical(logits=goal_logits)
            goals = dist.sample()  # (B, num_bees)
            log_probs = dist.log_prob(goals)  # (B, num_bees)
        else:
            probs = torch.softmax(goal_logits, dim=-1)
            goals = torch.argmax(probs, dim=-1)  # (B, num_bees)
            log_probs = torch.log(probs.gather(-1, goals.unsqueeze(-1)) + 1e-8).squeeze(-1)
        
        return goals, log_probs


class GoalConditionedWorker(nn.Module):
    """
    Low-level policy that executes actions conditioned on a goal.
    
    Extends the original Actor with goal conditioning.
    """
    
    def __init__(
        self,
        num_bees: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        num_flowers: int = 12,
        retask_board_size: int = 0,
        num_goals: int = 4,
        grid_size: int = 50,
    ):
        super().__init__()
        self.num_bees = num_bees
        self.action_dim = action_dim
        self.num_flowers = int(num_flowers)
        self.hidden_dim = hidden_dim
        self.retask_board_size = int(retask_board_size)
        self.num_goals = num_goals
        self.grid_size = float(grid_size)
        
        # --- Feature Encoders (same as original Actor) ---
        self.position_fc = nn.Sequential(nn.Linear(3, 64), nn.LayerNorm(64), nn.ReLU())
        self.status_fc = nn.Sequential(nn.Linear(2, 32), nn.LayerNorm(32), nn.ReLU())
        self.battery_fc = nn.Sequential(nn.Linear(2, 32), nn.LayerNorm(32), nn.ReLU())
        self.step_fc = nn.Sequential(nn.Linear(1, 32), nn.LayerNorm(32), nn.ReLU())
        self.cons_fc = nn.Sequential(nn.Linear(num_bees, 64), nn.LayerNorm(64), nn.ReLU())
        self.action_availability_fc = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.ReLU())
        
        # --- Goal Encoder (NEW) ---
        self.goal_fc = nn.Sequential(
            nn.Linear(num_goals, 32),  # One-hot goal encoding
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        # --- Flower Encoder with Attention ---
        self.flower_embed_dim = 128
        self.flowers_fc = nn.Sequential(
            nn.Linear(12, self.flower_embed_dim),
            nn.LayerNorm(self.flower_embed_dim),
            nn.ReLU(),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.flower_embed_dim,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        )
        self.flower_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Optional retask board encoder
        self.retask_board_fc = None
        retask_board_dim = 0
        if retask_board_size > 0:
            retask_input_dim = 5 * retask_board_size
            self.retask_board_fc = nn.Sequential(
                nn.Linear(retask_input_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )
            retask_board_dim = 128
        
        # Trunk input: original + battery + goal embedding
        # 64 + 32 + 32(battery) + 128 + 32 + 64 + retask_board_dim + 32 + 32 (goal)
        trunk_input = 64 + 32 + 32 + self.flower_embed_dim + 32 + 64 + retask_board_dim + 32 + 32
        
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        # Optional claim head
        self.claim_head = None
        if retask_board_size > 0:
            self.claim_head = nn.Linear(hidden_dim, retask_board_size + 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs_dict: dict, goal: torch.Tensor, return_claims: bool = False):
        """
        Args:
            obs_dict: Observation dict (same as original Actor)
            goal: (B,) goal index or (B, num_goals) one-hot encoding
        
        Returns:
            action_logits: (B, action_dim)
        """
        def ensure2d(x: torch.Tensor) -> torch.Tensor:
            x = x if x.dim() == 2 else x.view(1, -1)
            return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize inputs
        position = ensure2d(obs_dict["position"]) / self.grid_size
        status = ensure2d(obs_dict["status"])
        battery = ensure2d(obs_dict.get("battery", torch.zeros(1, 2, device=position.device)))
        
        # Flower attention
        flowers = ensure2d(obs_dict["flowers"])
        B = flowers.shape[0]
        
        flowers_reshaped = flowers.view(B, self.num_flowers, 12)
        flowers_reshaped[:, :, 0] /= self.grid_size
        flowers_reshaped[:, :, 1] /= self.grid_size
        flowers_reshaped[:, :, 2] /= 10.0
        
        flower_embed = self.flowers_fc(flowers_reshaped)
        attended_flowers = self.flower_transformer(flower_embed)
        flw_out = attended_flowers.mean(dim=1)
        
        step_count = ensure2d(obs_dict["step_count"])
        consensus = ensure2d(obs_dict["consensus"])
        
        # Goal encoding
        if goal.dim() == 1:
            # Convert to one-hot
            goal_onehot = torch.zeros(B, self.num_goals, device=goal.device)
            goal_onehot.scatter_(1, goal.unsqueeze(1), 1.0)
        else:
            goal_onehot = goal
        goal_out = self.goal_fc(goal_onehot)
        
        # Encode all features
        pos_out = self.position_fc(position)
        sta_out = self.status_fc(status)
        bat_out = self.battery_fc(battery)
        stp_out = self.step_fc(step_count)
        con_out = self.cons_fc(consensus)
        
        # Action availability
        act_avail = obs_dict.get("action_availability", None)
        if act_avail is not None:
            aa_out = self.action_availability_fc(ensure2d(act_avail))
        else:
            aa_out = torch.zeros(B, 32, device=pos_out.device, dtype=pos_out.dtype)
        
        # Retask board
        if self.retask_board_fc is not None:
            retask_board = ensure2d(obs_dict.get("retask_board", torch.zeros(1)))
            if retask_board.numel() > 0:
                rtb_out = self.retask_board_fc(retask_board)
            else:
                rtb_out = torch.zeros(B, 128, device=pos_out.device, dtype=pos_out.dtype)
        else:
            rtb_out = None
        
        # Concatenate all features including goal
        if rtb_out is not None:
            h = torch.cat([pos_out, sta_out, bat_out, flw_out, stp_out, con_out, rtb_out, aa_out, goal_out], dim=-1)
        else:
            h = torch.cat([pos_out, sta_out, bat_out, flw_out, stp_out, con_out, aa_out, goal_out], dim=-1)
        
        h = self.trunk(h)
        action_logits = self.policy_head(h)
        
        if return_claims and self.claim_head is not None:
            claim_logits = self.claim_head(h)
            return action_logits, claim_logits
        return action_logits


class HRLCritic(nn.Module):
    """
    Centralized critic for HRL that evaluates both manager and worker.
    
    Can be used for:
    - Manager value estimation (global state)
    - Worker value estimation (local state + goal)
    """
    
    def __init__(
        self,
        global_state_size: int,
        num_bees: int,
        num_goals: int = 4,
        hidden_dim: int = 512,
        grid_size: int = 50,
    ):
        super().__init__()
        self.global_state_size = int(global_state_size)
        self.num_bees = int(num_bees)
        self.num_goals = num_goals
        self.hidden_dim = hidden_dim
        self.grid_size = float(grid_size)
        
        # Manager critic (global state only)
        self.manager_trunk = nn.Sequential(
            nn.Linear(global_state_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.manager_v_head = nn.Linear(hidden_dim, 1)
        
        # Worker critic (global state + goals for all bees)
        worker_input = global_state_size + num_bees * num_goals
        self.worker_trunk = nn.Sequential(
            nn.Linear(worker_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.worker_v_head = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward_manager(self, global_state: torch.Tensor) -> torch.Tensor:
        """Value for manager policy."""
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        state_norm = global_state / self.grid_size
        state_norm = torch.nan_to_num(state_norm, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.manager_trunk(state_norm)
        return self.manager_v_head(h)
    
    def forward_worker(self, global_state: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Value for worker policy given goals."""
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        
        B = global_state.shape[0]
        
        # Convert goals to one-hot if needed
        if goals.dim() == 1:
            goals = goals.unsqueeze(0)
        if goals.shape[-1] != self.num_goals:
            # goals is (B, num_bees) indices, convert to one-hot
            goals_onehot = torch.zeros(B, self.num_bees, self.num_goals, device=goals.device)
            for i in range(self.num_bees):
                goals_onehot[:, i, :].scatter_(1, goals[:, i:i+1], 1.0)
            goals_flat = goals_onehot.view(B, -1)
        else:
            goals_flat = goals.view(B, -1)
        
        state_norm = global_state / self.grid_size
        state_norm = torch.nan_to_num(state_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        combined = torch.cat([state_norm, goals_flat], dim=-1)
        h = self.worker_trunk(combined)
        return self.worker_v_head(h)


# Helper functions for HRL
def build_manager_obs(env, device: str = "cpu") -> dict:
    """
    Build observation dict for manager from environment state.
    """
    num_bees = len(env.bees)
    num_flowers = len(env.flowers)
    
    # Bee states: position(3) + status(2) + battery_ratio + load_ratio + is_active
    bee_states = []
    for i, bee in enumerate(env.bees):
        battery_ratio = env._battery[i] / max(1.0, env._battery_max[i])
        load_ratio = bee.load / max(1.0, bee.capacity)
        is_active = 1.0 if not bee.truncated and env._recharge_until[i] <= env.steps else 0.0
        
        bee_state = [
            bee.fx / env.grid_size,
            bee.fy / env.grid_size,
            bee.fz / env.grid_size,
            float(bee.mode) / 2.0,
            battery_ratio,
            load_ratio,
            is_active,
        ]
        bee_states.append(bee_state)
    
    # Flower states: same 12 features as worker
    flowers = []
    for f in env.flowers:
        x = (f.x + 0.5) / env.grid_size
        y = (f.y + 0.5) / env.grid_size
        pol = f.pollen / max(1.0, env.bee_capacity)
        harvested = 1.0 if f.harvested else 0.0
        has_assignment = 1.0 if f.assigned_bee is not None else 0.0
        is_harvestable = 1.0 if f.is_harvestable_at_time(env.steps) else 0.0
        is_hard = 1.0 if f.window_type == "HARD" else 0.0
        time_to_window = f.time_until_next_window(env.steps) / max(1.0, env.max_steps)
        
        flower_state = [x, y, pol, harvested, has_assignment, 0.0, 0.0, 0.0, 0.0, is_harvestable, is_hard, time_to_window]
        flowers.append(flower_state)
    
    # Context
    step_norm = env.steps / max(1.0, env.max_steps)
    harvest_ratio = sum(1 for f in env.flowers if f.harvested) / max(1, num_flowers)
    active_ratio = sum(1 for i in range(num_bees) if not env.bees[i].truncated) / max(1, num_bees)
    context = [step_norm, harvest_ratio, active_ratio]
    
    return {
        "bee_states": torch.tensor([bee_states], dtype=torch.float32, device=device),
        "flowers": torch.tensor([flowers], dtype=torch.float32, device=device),
        "context": torch.tensor([context], dtype=torch.float32, device=device),
    }


GOAL_NAMES = {
    ManagerPolicy.GOAL_IDLE: "IDLE",
    ManagerPolicy.GOAL_HARVEST: "HARVEST",
    ManagerPolicy.GOAL_GROOM: "GROOM",
    ManagerPolicy.GOAL_ASSIST: "ASSIST",
}
