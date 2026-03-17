"""
Physics-based gossip relay for orbital bee communication.

Replaces the old chain relay v3 with line-of-sight, range-limited,
hop-by-hop message passing.  Each GossipMessage carries an orphan task
(flower_id) and propagates one hop per env step.

Integration:
    from gossiper import Gossiper

    # in __init__:
    self.gossiper = Gossiper(env=self)

    # in reset():
    self.gossiper.reset(self.num_bees)

    # when a bee dies (battery=0):
    self.gossiper.broadcast_tasks(dead_bee_id, flower_ids, step)

    # every step (replaces _propagate_retask_board):
    accepted = self.gossiper.propagate(step)

    # in _get_observations (replaces bee.retask_board reading):
    retask_feat = self.gossiper.get_retask_obs(bee_id, board_size)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bees_env import BeeForagingEnv


# ── GossipMessage ────────────────────────────────────────────
@dataclass
class GossipMessage:
    """Single orphan-task message traveling the gossip network."""

    flower_id: int
    priority: float
    origin_bee: int          # bee that died / released the task
    sender_bee: int          # last bee that forwarded this message
    x: float                 # flower grid x
    y: float                 # flower grid y
    pollen: float
    window_type: str         # HARD / SOFT / NONE
    window_start: int
    window_end: int
    ttl: int = 15            # max hops before message expires
    hops: int = 0            # hops so far
    age: int = 0             # steps since creation
    max_age: int = 200       # steps before message expires
    created_step: int = 0
    seen_by: set = field(default_factory=set)
    claimed_by: int | None = None   # bee that accepted the task (None = unclaimed)

    @property
    def expired(self) -> bool:
        return self.hops >= self.ttl or self.age >= self.max_age

    def tick(self):
        """Advance one env step (call once per step, not per hop)."""
        self.age += 1


# ── Gossiper ─────────────────────────────────────────────────
class Gossiper:
    """
    Physics-based gossip relay.

    Communication rules:
        1. Range-limited:  two bees must be within ``max_range`` grid units.
        2. Line-of-sight:  no Earth occlusion (ray-sphere test).  Disabled
           when ``earth_radius <= 0`` (flat-grid mode).
        3. One hop per step:  messages advance to the nearest *unseen*,
           *reachable* bee each env step.
        4. Evaluation on arrival:  receiving bee decides accept / reject
           based on reachability, capacity, deadline, and workload.
    """

    def __init__(
        self,
        env: "BeeForagingEnv",
        max_range: float | None = None,
        max_hops: int | None = None,
        max_age: int = 200,
        cooldown_steps: int = 3,
        earth_radius: float = 0.0,
    ):
        self.env = env
        self.max_range = max_range if max_range is not None else env.harvest_radius * 1.5
        self.max_hops = max_hops if max_hops is not None else env.num_bees
        self.max_age = max_age
        self.cooldown_steps = cooldown_steps
        self.earth_radius = earth_radius

        # Per-bee inbox: list[GossipMessage]
        self.inboxes: list[list[GossipMessage]] = []
        # Per-bee cooldown timer (steps until bee can forward again)
        self._cooldown: np.ndarray = np.array([], dtype=int)
        # Tracking
        self.total_messages_sent = 0
        self.total_messages_accepted = 0
        self.total_messages_expired = 0

    # ── lifecycle ────────────────────────────────────────────
    def reset(self, num_bees: int):
        self.inboxes = [[] for _ in range(num_bees)]
        self._cooldown = np.zeros(num_bees, dtype=int)
        self.total_messages_sent = 0
        self.total_messages_accepted = 0
        self.total_messages_expired = 0

    # ── broadcast (called when a bee dies) ───────────────────
    def broadcast_tasks(self, dead_bee: int, flower_ids: list[int], step: int):
        """
        Create GossipMessages for every unharvested flower that ``dead_bee``
        owned and place them in the nearest reachable bee's inbox.
        """
        env = self.env
        bee = env.bees[dead_bee]

        for fj in flower_ids:
            if fj >= len(env.flowers):
                continue
            f = env.flowers[fj]
            if f.harvested:
                continue

            # Unassign flower from dead bee
            f.assigned_bee = None

            msg = GossipMessage(
                flower_id=fj,
                priority=f.priority,
                origin_bee=dead_bee,
                sender_bee=dead_bee,
                x=f.x,
                y=f.y,
                pollen=f.pollen,
                window_type=f.window_type,
                window_start=f.window_start,
                window_end=f.window_end,
                ttl=self.max_hops,
                max_age=self.max_age,
                created_step=step,
                seen_by={dead_bee},
            )

            # deliver to nearest reachable active bee
            target = self._nearest_reachable(dead_bee, msg.seen_by, step)
            if target is not None:
                msg.hops += 1
                msg.sender_bee = dead_bee
                self.inboxes[target].append(msg)
                self.total_messages_sent += 1
                if env.verbose:
                    print(
                        f"[GOSSIP] Bee {dead_bee} died → flower {fj} "
                        f"sent to bee {target}"
                    )
            else:
                # No reachable bee now — keep flower unassigned in pool
                if env.verbose:
                    print(
                        f"[GOSSIP] Bee {dead_bee} died → flower {fj} "
                        f"released to pool (no reachable bee)"
                    )

    # ── per-step propagation ─────────────────────────────────
    def propagate(self, step: int, learned_claims: dict[int, int | None] | None = None) -> list[tuple[int, int]]:
        """
        Advance all messages one hop.  Returns list of (bee_id, flower_id)
        pairs for accepted tasks so the env can update assignments.

        Args:
            step: current env step
            learned_claims: optional dict {bee_id: inbox_slot_index_or_None}.
                When provided, the policy's claim decision overrides the
                hardcoded ``_evaluate_task`` heuristic.  ``None`` or an index
                equal to ``board_size`` means "don't claim anything".
                The slot indices refer to the priority-sorted inbox (same
                order exposed via ``get_retask_obs``).
        """
        env = self.env
        accepted: list[tuple[int, int]] = []
        use_learned = learned_claims is not None

        # Tick cooldowns
        self._cooldown = np.maximum(0, self._cooldown - 1)

        for i in range(len(self.inboxes)):
            bee = env.bees[i]
            if bee.truncated or env._recharge_until[i] > step:
                continue  # skip dead / recharging bees

            # Sort inbox by priority (same order as get_retask_obs)
            self.inboxes[i] = sorted(self.inboxes[i], key=lambda m: -m.priority)

            # Determine which slot (if any) this bee claims via the policy
            claimed_slot: int | None = None
            if use_learned and i in learned_claims:
                raw = learned_claims[i]
                board_size = getattr(env, "retask_board_size", 3)
                # slot == board_size or None means "no claim"
                if raw is not None and 0 <= raw < board_size and raw < len(self.inboxes[i]):
                    claimed_slot = raw

            new_inbox: list[GossipMessage] = []
            for slot_idx, msg in enumerate(self.inboxes[i]):
                msg.tick()

                # ── expiry checks ────────────────────────────
                if msg.expired:
                    self.total_messages_expired += 1
                    continue
                fj = msg.flower_id
                if fj >= len(env.flowers):
                    continue
                f = env.flowers[fj]
                if f.harvested:
                    continue
                if f.window_type == "HARD" and step > f.window_end:
                    self.total_messages_expired += 1
                    continue
                if f.assigned_bee is not None:
                    # Someone else already claimed it
                    continue

                # ── evaluate: should this bee accept? ────────
                if i not in msg.seen_by:
                    msg.seen_by.add(i)

                    if use_learned:
                        # Use the policy's claim decision
                        should_accept = (claimed_slot == slot_idx)
                    else:
                        # Fallback: hardcoded heuristic
                        should_accept = self._evaluate_task(i, msg, step)

                    if should_accept:
                        # ACCEPT
                        f.assigned_bee = i
                        if fj not in bee.assigned_flowers:
                            bee.assigned_flowers.append(fj)
                        if i < len(env.assignments):
                            if fj not in env.assignments[i]:
                                env.assignments[i].append(fj)
                        msg.claimed_by = i
                        accepted.append((i, fj))
                        self.total_messages_accepted += 1
                        # Remove this flower from ALL inboxes
                        self._clear_flower_from_inboxes(fj)
                        if env.verbose:
                            print(
                                f"[GOSSIP] Bee {i} ACCEPTS flower {fj} "
                                f"(hops={msg.hops}, origin={msg.origin_bee})"
                            )
                        continue  # don't keep in inbox after accept
                    else:
                        # REJECT — forward to nearest unseen reachable bee
                        if self._cooldown[i] == 0:
                            target = self._nearest_reachable(i, msg.seen_by, step)
                            if target is not None:
                                msg.hops += 1
                                msg.sender_bee = i
                                self.inboxes[target].append(msg)
                                self._cooldown[i] = self.cooldown_steps
                                self.total_messages_sent += 1
                                if env.verbose:
                                    print(
                                        f"[GOSSIP] Bee {i} → Bee {target}: "
                                        f"forwarded flower {fj} (hops={msg.hops})"
                                    )
                                continue  # forwarded, don't keep
                            else:
                                # No unseen reachable bee — keep in inbox for next step
                                new_inbox.append(msg)
                        else:
                            # On cooldown — keep for next step
                            new_inbox.append(msg)
                else:
                    # Already seen by this bee — try to forward
                    if self._cooldown[i] == 0:
                        target = self._nearest_reachable(i, msg.seen_by, step)
                        if target is not None:
                            msg.hops += 1
                            msg.sender_bee = i
                            self.inboxes[target].append(msg)
                            self._cooldown[i] = self.cooldown_steps
                            self.total_messages_sent += 1
                            continue
                        else:
                            new_inbox.append(msg)
                    else:
                        new_inbox.append(msg)

            self.inboxes[i] = new_inbox

        return accepted

    # ── observation helper ───────────────────────────────────
    def get_retask_obs(self, bee_id: int, board_size: int) -> list[float]:
        """
        Return a flat feature vector for the gossip inbox of ``bee_id``.
        Each slot: [x_norm, y_norm, pollen_norm, can_fit, hops_norm]
        Matches the old retask_board observation shape exactly.
        """
        env = self.env
        bee = env.bees[bee_id]
        max_pollen = float(max(1, env.bee_capacity))
        feat: list[float] = []

        # Sort inbox by priority desc
        inbox = sorted(self.inboxes[bee_id], key=lambda m: -m.priority)

        for msg in inbox[:board_size]:
            fj = msg.flower_id
            if fj < 0 or fj >= len(env.flowers):
                feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            f = env.flowers[fj]
            can_fit = 1.0 if (bee.load + f.pollen) <= bee.capacity + 1e-9 else 0.0
            feat.extend([
                (f.x + 0.5) / env.grid_size,           # x normalized
                (f.y + 0.5) / env.grid_size,           # y normalized
                float(f.pollen) / max_pollen,           # pollen normalized
                can_fit,                                # can fit in capacity
                float(msg.hops) / max(1.0, float(self.max_hops)),  # hops normalized
            ])

        # Pad to fixed size
        expected = 5 * board_size
        while len(feat) < expected:
            feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return feat

    # ── physics helpers ──────────────────────────────────────
    def _can_communicate(self, bee_a: int, bee_b: int) -> bool:
        """Check range + line-of-sight between two bees."""
        env = self.env
        a = env.bees[bee_a]
        b = env.bees[bee_b]

        dx = a.fx - b.fx
        dy = a.fy - b.fy
        dz = a.fz - b.fz
        dist = math.sqrt(dx * dx + dy * dy + env.lambda_z * dz * dz)

        if dist > self.max_range:
            return False

        if self.earth_radius > 0:
            return self._line_of_sight(
                np.array([a.fx, a.fy, a.fz]),
                np.array([b.fx, b.fy, b.fz]),
            )
        return True

    def _line_of_sight(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Ray-sphere intersection test for Earth occlusion.
        Returns True if the line segment p1→p2 does NOT intersect the
        sphere centered at grid_center with radius ``earth_radius``.
        """
        env = self.env
        center = np.array([env.grid_size / 2.0, env.grid_size / 2.0, 0.0])
        d = p2 - p1
        f = p1 - center
        r = self.earth_radius

        a = float(np.dot(d, d))
        b = 2.0 * float(np.dot(f, d))
        c = float(np.dot(f, f)) - r * r

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return True  # no intersection

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a + 1e-12)
        t2 = (-b + sqrt_disc) / (2.0 * a + 1e-12)

        # Check if intersection is within the segment [0, 1]
        if t1 > 1.0 or t2 < 0.0:
            return True
        return False

    def _nearest_reachable(
        self, from_bee: int, seen_by: set, step: int
    ) -> int | None:
        """Find the nearest active, unseen, in-range bee."""
        env = self.env
        bee = env.bees[from_bee]
        best_id = None
        best_d = float("inf")

        for j in range(len(env.bees)):
            if j == from_bee or j in seen_by:
                continue
            other = env.bees[j]
            if other.truncated or env._recharge_until[j] > step:
                continue
            if not self._can_communicate(from_bee, j):
                continue

            dx = other.fx - bee.fx
            dy = other.fy - bee.fy
            dz = other.fz - bee.fz
            d = math.sqrt(dx * dx + dy * dy + env.lambda_z * dz * dz)
            if d < best_d:
                best_d = d
                best_id = j

        return best_id

    def _evaluate_task(self, bee_id: int, msg: GossipMessage, step: int) -> bool:
        """Decide whether bee should accept this orphan task."""
        env = self.env
        bee = env.bees[bee_id]
        fj = msg.flower_id
        f = env.flowers[fj]
        cx, cy = f.center_xy

        # 1. REACHABILITY — orbit distance
        orbit_dist = env._min_distance_to_orbit(bee, cx, cy)
        can_reach = orbit_dist <= (env.harvest_radius + env.reach_margin)

        # 2. CAPACITY
        can_fit = (bee.load + f.pollen) <= bee.capacity + 1e-9

        # 3. DEADLINE (HARD windows only)
        can_meet_deadline = True
        if f.window_type == "HARD":
            steps_remaining = f.window_end - step
            if steps_remaining < 10:
                can_meet_deadline = False

        # 4. WORKLOAD — don't overload a single bee
        current_workload = sum(
            1 for fk in bee.assigned_flowers
            if fk < len(env.flowers) and not env.flowers[fk].harvested
        )
        workload_ok = current_workload < 5

        return can_reach and can_fit and can_meet_deadline and workload_ok

    def _clear_flower_from_inboxes(self, flower_id: int):
        """Remove all messages about ``flower_id`` from every inbox."""
        for inbox in self.inboxes:
            inbox[:] = [m for m in inbox if m.flower_id != flower_id]
