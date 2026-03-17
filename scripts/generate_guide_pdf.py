#!/usr/bin/env python3
"""
Generate a comprehensive PDF guide for the Bee_Fi project.
Uses fpdf2 for PDF generation with detailed formatting.
"""

from fpdf import FPDF
import os
import datetime


class BeeFiGuide(FPDF):
    """Custom PDF class for Bee_Fi documentation."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        # Track section numbering
        self._section_num = 0
        self._subsection_num = 0

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "Bee_Fi  |  Multi-Agent RL for Orbital Satellite Task Scheduling", align="L")
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, 14, 200, 14)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Bee_Fi Project Documentation  |  March 2026  |  Confidential", align="C")

    def title_page(self):
        self.add_page()
        self.ln(50)
        # Title
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(20, 60, 120)
        self.cell(0, 15, "Bee_Fi", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        self.set_font("Helvetica", "", 18)
        self.set_text_color(60, 60, 60)
        self.cell(0, 10, "Multi-Agent Reinforcement Learning for", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 10, "Orbital Satellite Task Scheduling", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(10)
        # Subtitle line
        self.set_draw_color(20, 60, 120)
        self.set_line_width(0.8)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(10)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "Comprehensive System Guide", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        self.cell(0, 8, "With Basilisk Orbital Dynamics Integration", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(20)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 7, f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 7, "Version: 2.0 (BSK Integration)", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 7, "Platform: PettingZoo + PyTorch + Basilisk", align="C", new_x="LMARGIN", new_y="NEXT")

    def section(self, title):
        self._section_num += 1
        self._subsection_num = 0
        self.ln(4)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 60, 120)
        self.cell(0, 10, f"{self._section_num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def subsection(self, title):
        self._subsection_num += 1
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(40, 80, 140)
        self.cell(0, 8, f"{self._section_num}.{self._subsection_num} {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsubsection(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 100, 160)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(indent, 5.5, "")
        self.cell(5, 5.5, "-")
        self.multi_cell(170 - indent, 5.5, text)

    def bold_bullet(self, label, text, indent=10):
        self.set_text_color(30, 30, 30)
        self.cell(indent, 5.5, "")
        self.set_font("Helvetica", "B", 10)
        lw = self.get_string_width(f"{label}: ") + 2
        self.cell(lw, 5.5, f"{label}: ")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(170 - indent - lw, 5.5, text)

    def code_block(self, text, width=190):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(30, 30, 30)
        self.set_draw_color(200, 200, 210)
        x = self.get_x()
        y = self.get_y()
        lines = text.split("\n")
        line_h = 4.5
        block_h = len(lines) * line_h + 6
        # Check if we need a page break
        if y + block_h > self.h - 25:
            self.add_page()
            y = self.get_y()
        self.rect(x, y, width, block_h, "DF")
        self.set_xy(x + 3, y + 3)
        for line in lines:
            self.set_x(x + 3)
            self.cell(width - 6, line_h, line, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = 190 / len(headers)
            col_widths = [w] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        alt = False
        for row in rows:
            if alt:
                self.set_fill_color(245, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            max_h = 7
            for i, cell_text in enumerate(row):
                self.cell(col_widths[i], max_h, str(cell_text), border=1, fill=True, align="L")
            self.ln()
            alt = not alt
        self.ln(2)

    def key_value_block(self, items):
        """Render a key-value pair list."""
        for key, value in items:
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(40, 40, 40)
            self.cell(55, 6, key + ":")
            self.set_font("Helvetica", "", 10)
            self.set_text_color(60, 60, 60)
            self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def info_box(self, title, text):
        """Colored info box."""
        self.set_fill_color(230, 240, 255)
        self.set_draw_color(20, 60, 120)
        self.set_line_width(0.3)
        y = self.get_y()
        lines = text.split("\n")
        h = len(lines) * 5.5 + 14
        if y + h > self.h - 25:
            self.add_page()
            y = self.get_y()
        self.rect(10, y, 190, h, "DF")
        self.set_xy(13, y + 3)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(20, 60, 120)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_x(13)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        for line in lines:
            self.set_x(13)
            self.cell(0, 5.5, line, new_x="LMARGIN", new_y="NEXT")
        self.set_y(y + h + 3)


def build_guide():
    pdf = BeeFiGuide()
    pdf.alias_nb_pages()

    # ─── TITLE PAGE ───
    pdf.title_page()

    # ─── TABLE OF CONTENTS ───
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "Table of Contents", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    toc_items = [
        ("1.", "System Overview", "3"),
        ("2.", "Architecture", "4"),
        ("3.", "Neural Network Design", "6"),
        ("4.", "Environment (BeeForagingEnv)", "8"),
        ("5.", "Basilisk Integration", "12"),
        ("6.", "Task System & Metadata", "15"),
        ("7.", "Gossip Communication Protocol", "17"),
        ("8.", "Training Pipeline", "18"),
        ("9.", "BSK Evaluator & Telemetry", "21"),
        ("10.", "Configuration Reference", "23"),
        ("11.", "Performance Results", "24"),
        ("12.", "File Reference", "25"),
        ("13.", "CLI Commands Reference", "26"),
        ("14.", "Known Issues & Future Work", "27"),
    ]
    for num, title, pg in toc_items:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(20, 60, 120)
        pdf.cell(12, 7, num)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(140, 7, title)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 7, pg, align="R", new_x="LMARGIN", new_y="NEXT")

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: SYSTEM OVERVIEW
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("System Overview")

    pdf.body_text(
        "Bee_Fi is a Multi-Agent Reinforcement Learning (MARL) system that simulates "
        "orbital satellite swarms performing coordinated task harvesting. The system uses "
        "a bio-inspired metaphor where satellites are modeled as 'bees' and observation "
        "tasks are modeled as 'flowers' with pollen values representing task priority."
    )
    pdf.ln(2)
    pdf.body_text(
        "The project combines orbital mechanics (Keplerian, SGP4, and Basilisk high-fidelity "
        "simulation), battery-driven fault tolerance, time-constrained task windows, "
        "decentralized gossip-based communication, and PPO-trained transformer attention "
        "policies for decision-making."
    )

    pdf.subsection("Key Capabilities")
    pdf.bold_bullet("Orbital Task Scheduling", "25 satellites coordinate to complete 50 observation tasks on passive orbital trajectories")
    pdf.bold_bullet("Battery Fault Tolerance", "When a satellite's battery depletes, its tasks are automatically redistributed via gossip relay")
    pdf.bold_bullet("Time Window Constraints", "Tasks have HARD (one-time, severe penalty), SOFT (repeating), and NONE (anytime) windows")
    pdf.bold_bullet("Basilisk Integration", "High-fidelity orbital dynamics via NASA Basilisk framework with J2 perturbations")
    pdf.bold_bullet("Decentralized Communication", "Physics-based gossip protocol with range-limited, hop-by-hop message passing")
    pdf.bold_bullet("Task Expiration", "Priority-scaled deadlines; expired tasks incur penalties and are removed from assignments")

    pdf.subsection("Technology Stack")
    pdf.table(
        ["Component", "Technology", "Version"],
        [
            ["RL Framework", "PettingZoo (ParallelEnv)", ">= 1.24.0"],
            ["Deep Learning", "PyTorch", ">= 2.0.0"],
            ["Orbital Dynamics", "Basilisk (BSK)", "Built from source"],
            ["Orbital Propagation", "SGP4 (optional)", ">= 2.20"],
            ["Training Algorithm", "PPO (Proximal Policy Optimization)", "Custom impl"],
            ["Visualization", "Matplotlib 3D + Dash", ">= 3.5.0"],
            ["Configuration", "YAML", "PyYAML >= 6.0"],
            ["Logging", "TensorBoard", ">= 2.12.0"],
        ],
        col_widths=[45, 85, 60],
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: ARCHITECTURE
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Architecture")

    pdf.body_text(
        "The system follows a modular architecture with clear separation between the "
        "simulation environment, neural network policies, training loop, and evaluation pipeline."
    )

    pdf.subsection("High-Level Architecture")
    pdf.code_block(
        "+-------------------+     +-------------------+     +------------------+\n"
        "|  Training Loop    |     |   Environment     |     |  Evaluation      |\n"
        "|  train_orbital_   |<--->|   BeeForagingEnv  |<--->|  bsk_evaluator   |\n"
        "|  v2.py            |     |   bees_env.py     |     |  .py             |\n"
        "+--------+----------+     +--------+----------+     +------------------+\n"
        "         |                         |                                     \n"
        "    +----v----+          +---------v---------+     +------------------+  \n"
        "    |  Actor  |          |   BSKInterface    |     |  Gossiper        |  \n"
        "    |  Critic |          |   bsk_interface   |     |  gossiper.py     |  \n"
        "    |  (policy|          |   .py             |     |  (comms relay)   |  \n"
        "    +---------+          +-------------------+     +------------------+  \n"
        "                                 |                                       \n"
        "                         +-------v--------+                              \n"
        "                         |   Basilisk     |                              \n"
        "                         |   Simulator    |                              \n"
        "                         +----------------+                              "
    )

    pdf.subsection("Data Flow")
    pdf.body_text("Each simulation step follows this pipeline:")
    pdf.ln(1)
    pdf.bold_bullet("Step 1", "BSKInterface advances Basilisk by 1 second, returns ECI positions + battery for each satellite")
    pdf.bold_bullet("Step 2", "bees_env.py maps ECI positions to grid coordinates via centroid-centering and meters_per_unit scaling")
    pdf.bold_bullet("Step 3", "Environment syncs battery from BSK state, drains per-step as needed")
    pdf.bold_bullet("Step 4", "Per-bee observations are built: position, battery, flower features (12 per flower), consensus, retask board")
    pdf.bold_bullet("Step 5", "Actor network receives observations, outputs action (0=DONOTHING, 1=HARVEST, 2=GROOM) + claim slot")
    pdf.bold_bullet("Step 6", "Environment executes actions, computes rewards, checks task expiration, propagates gossip")
    pdf.bold_bullet("Step 7", "Updated observations returned to policy for next step")

    pdf.subsection("Operating Modes")
    pdf.table(
        ["Mode", "Dynamics", "Use Case"],
        [
            ["Keplerian", "Analytical 2-body orbits", "Fast training (default)"],
            ["SGP4", "TLE-based propagation", "Real satellite elements"],
            ["Basilisk (BSK)", "High-fidelity N-body + J2", "Evaluation & BSK training"],
        ],
        col_widths=[40, 70, 80],
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: NEURAL NETWORK DESIGN
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Neural Network Design")

    pdf.subsection("Actor Network (Per-Agent Policy)")
    pdf.body_text(
        "The Actor is a transformer-attention-based policy network. Each bee receives its "
        "own local observation and independently selects an action. The architecture uses "
        "learned attention over flower features to focus on the most relevant tasks."
    )

    pdf.subsubsection("Feature Encoders (Input Stage)")
    pdf.table(
        ["Input", "Raw Dim", "Encoder", "Output Dim"],
        [
            ["Position (x,y,z)", "3", "Linear + LayerNorm + ReLU", "64"],
            ["Status (load, capacity)", "2", "Linear + LayerNorm + ReLU", "32"],
            ["Flowers (N x 12 features)", "12 per flower", "Linear + LayerNorm + ReLU", "128 per flower"],
            ["Step count", "1", "Linear + LayerNorm + ReLU", "32"],
            ["Consensus (last actions)", "N_bees", "Linear + LayerNorm + ReLU", "64"],
            ["Retask board (M x 5)", "5*M", "Linear + LayerNorm + ReLU", "128"],
            ["Action availability", "3", "Linear + LayerNorm + ReLU", "32"],
        ],
        col_widths=[50, 25, 65, 30],
    )

    pdf.subsubsection("Flower Attention (Transformer)")
    pdf.body_text(
        "Each flower's 12-feature vector is embedded to 128 dimensions, then processed "
        "by a 1-layer Transformer Encoder with 4 attention heads. The output is mean-pooled "
        "to produce a single 128-dimensional flower summary vector. This allows the agent "
        "to attend to the most relevant flowers (nearby, high-priority, approaching deadline)."
    )

    pdf.subsubsection("Trunk Network")
    pdf.body_text(
        "All encoded features are concatenated (total ~480 dims) and passed through "
        "two Linear layers: 480 -> 256 -> 256, each with LayerNorm and ReLU activation."
    )

    pdf.subsubsection("Output Heads")
    pdf.table(
        ["Head", "Output", "Description"],
        [
            ["Policy Head", "3 logits", "DONOTHING / HARVEST / GROOM probabilities"],
            ["Claim Head", "M+1 logits", "Retask board slot selection (0..M-1) or no-claim"],
        ],
        col_widths=[40, 40, 110],
    )

    pdf.subsection("Centralized Critic Network")
    pdf.body_text(
        "The critic observes the global state (all bees concatenated + all flower features) "
        "to estimate state value V(s). This centralized training / decentralized execution "
        "(CTDE) paradigm gives the critic a full view while each actor only sees its local obs."
    )
    pdf.ln(1)
    pdf.body_text(
        "Architecture: Global state -> Linear(N*obs_dim, 512) + LayerNorm + ReLU -> "
        "Linear(512, 512) + LayerNorm + ReLU -> Linear(512, 1) -> V(s) scalar."
    )

    pdf.info_box(
        "Key Design Decision",
        "At inference time, ONLY the Actor is used. The Critic is discarded.\n"
        "This means deployed satellites only need the lightweight Actor network,\n"
        "not the large Critic that sees global state."
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: ENVIRONMENT
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Environment (BeeForagingEnv)")

    pdf.body_text(
        "BeeForagingEnv is a PettingZoo ParallelEnv subclass implementing the full "
        "multi-agent satellite task scheduling simulation. It is approximately 2,220 lines "
        "of Python code and handles orbital mechanics, battery management, task assignment, "
        "time windows, gossip communication, and reward computation."
    )

    pdf.subsection("Agents (Bees / Satellites)")
    pdf.table(
        ["Property", "Default", "Description"],
        [
            ["num_bees", "25", "Number of satellites in the constellation"],
            ["Position", "(fx, fy, fz)", "3D grid coordinates (mapped from orbits or BSK)"],
            ["Battery", "450-750 steps", "Randomized per episode; drains 1.0/step"],
            ["Load / Capacity", "0.0 / 10.0", "Current pollen (task data) vs max capacity"],
            ["Orbital Elements", "a, e, i, Om, w", "Keplerian elements (synced from BSK in BSK mode)"],
            ["Mode", "IDLE/HARVEST/GROOM", "Current activity state"],
            ["Assigned Flowers", "list[int]", "Task queue assigned to this satellite"],
        ],
        col_widths=[40, 35, 115],
    )

    pdf.subsection("Tasks (Flowers)")
    pdf.table(
        ["Property", "Type", "Description"],
        [
            ["Position", "(x, y)", "Fixed on grid, placed along orbital paths"],
            ["Pollen", "float 0.8-10.0", "Task value / priority weight"],
            ["Priority", "float 0-1", "pollen / max_capacity"],
            ["Time Window", "HARD/SOFT/NONE", "Scheduling constraint type"],
            ["task_id", "str", "Unique ID: TASK-{id:03d}-{x}-{y}"],
            ["task_description", "str", "One of 5 task types (imagery, relay, etc.)"],
            ["status", "str", "unassigned / assigned / completed / expired"],
            ["deadline_step", "int", "Absolute step deadline (priority-scaled)"],
            ["created_step", "int", "Step when task was created"],
        ],
        col_widths=[40, 40, 110],
    )

    pdf.subsection("Action Space")
    pdf.table(
        ["Action", "ID", "Effect", "Reward"],
        [
            ["DONOTHING", "0", "Continue orbiting", "0 (small penalty if flowers nearby)"],
            ["HARVEST", "1", "Collect pollen from nearest valid flower", "+5.0 + pollen bonus (up to +10)"],
            ["GROOM", "2", "Offload pollen or recharge battery", "+5.0 to +15.0 (strategic)"],
        ],
        col_widths=[35, 15, 75, 65],
    )

    pdf.subsection("Observation Space (Per Bee)")
    pdf.body_text("Each bee receives a dictionary observation with these components:")
    pdf.table(
        ["Field", "Dimension", "Source"],
        [
            ["position", "3", "(x,y,z) / grid_size"],
            ["status", "2", "[load/capacity, capacity_norm]"],
            ["battery", "2", "[battery_frac, is_recharging]"],
            ["flowers", "N_flowers x 12", "12 features per flower (see below)"],
            ["step_count", "1", "current_step / max_steps"],
            ["consensus", "N_bees", "Last actions of all bees / 2.0"],
            ["retask_board", "M x 5", "Orphan task features from gossiper"],
            ["action_availability", "3", "[can_harvest, can_groom, can_idle]"],
        ],
        col_widths=[45, 35, 110],
    )

    pdf.subsubsection("Per-Flower Features (12 each)")
    pdf.table(
        ["#", "Feature", "Description"],
        [
            ["0", "x", "Normalized grid x position"],
            ["1", "y", "Normalized grid y position"],
            ["2", "pollen", "Pollen amount / max capacity"],
            ["3", "harvested", "1.0 if already harvested"],
            ["4", "mine", "1.0 if assigned to this bee"],
            ["5", "busy", "1.0 if another bee is currently harvesting"],
            ["6", "reachable", "1.0 if on this bee's reachable orbital path"],
            ["7", "fits", "1.0 if pollen fits in remaining capacity"],
            ["8", "dist_norm", "3D distance / harvest_radius"],
            ["9", "harvestable_now", "1.0 if within active time window"],
            ["10", "hard_window", "1.0 if HARD time window type"],
            ["11", "time_to_window", "Steps until next window / max_steps"],
        ],
        col_widths=[10, 40, 140],
    )

    pdf.subsection("Battery & Recharge System")
    pdf.body_text(
        "Each bee has a randomized battery capacity (450-750 steps). Battery drains by 1.0 "
        "per step. When battery reaches 0, the bee 'dies' and enters recharge mode for 40 "
        "steps. During recharge, all assigned flowers are released to the unassigned pool "
        "and broadcast via gossip relay."
    )
    pdf.ln(2)
    pdf.body_text(
        "Curriculum learning: 5% of episodes use 'low-battery mode' with 1/3 capacity and "
        "2-3x drain rate, forcing the policy to learn proactive recharge behavior."
    )

    pdf.subsection("Proactive Groom Mechanism")
    pdf.body_text(
        "Before processing a HARVEST action, the environment checks if the bee's remaining "
        "capacity can fit the smallest reachable flower. If not, it automatically triggers "
        "a GROOM action instead, preventing wasted harvest attempts on full-capacity bees."
    )

    pdf.subsection("Task Assignment & Retasking")
    pdf.bold_bullet("Initial Assignment", "Round-robin by descending pollen value at episode start")
    pdf.bold_bullet("Exclusivity", "One bee per flower (strict exclusion). Closest bee wins conflicts.")
    pdf.bold_bullet("Pool Claiming", "Unassigned flowers are exposed via retask board for nearby bees to claim")
    pdf.bold_bullet("Gossip Relay", "Dead bee's orphan tasks broadcast hop-by-hop to nearest active satellites")

    pdf.subsection("Episode Termination")
    pdf.body_text("An episode ends when any of these conditions is met:")
    pdf.bullet("All flowers are DONE (harvested or expired)")
    pdf.bullet("All bees are truncated AND no unassigned flowers remain")
    pdf.bullet("Maximum steps reached (1200)")
    pdf.ln(1)
    pdf.body_text(
        "End states: SUCCESS (all harvested), PARTIAL (mix of harvested and expired), TIMEOUT (max steps)."
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: BASILISK INTEGRATION
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Basilisk Integration")

    pdf.body_text(
        "Basilisk (BSK) is a NASA-developed astrodynamics framework providing high-fidelity "
        "orbital simulation. Bee_Fi integrates BSK via the BSKInterface wrapper class, "
        "replacing the analytical Keplerian propagator with a full N-body + J2 perturbation model."
    )

    pdf.subsection("BSKInterface (bsk_interface.py)")
    pdf.body_text("Thin wrapper (~420 lines) around multi-satellite Basilisk simulation. Creates N spacecraft with:")
    pdf.bold_bullet("Gravity Model", "Earth central body with J2 perturbations")
    pdf.bold_bullet("Battery", "SimpleBattery - 200 Wh capacity, 80% initial charge")
    pdf.bold_bullet("Power Sink", "SimplePowerSink - 3W constant draw")
    pdf.bold_bullet("Solar Panel", "SimpleSolarPanel - 0.32 m^2, 35% efficiency (optional)")
    pdf.bold_bullet("Navigation", "SimpleNav for clean position output")
    pdf.ln(2)
    pdf.body_text(
        "Default constellation: Walker-delta LEO at 550 km altitude, 53 degrees inclination, "
        "3 orbital planes. This matches typical communication satellite constellations."
    )

    pdf.subsection("BSK Mode in BeeForagingEnv")

    pdf.subsubsection("Reset (Initialization)")
    pdf.body_text("When use_basilisk=True, the reset() method performs these additional steps:")
    pdf.bullet("BSKInterface creates N spacecraft in Basilisk simulator")
    pdf.bullet("get_positions_grid() maps ECI metres to grid coords (centroid-centered, scaled by meters_per_unit=500,000)")
    pdf.bullet("Kepler element sync: Bee orbital elements (a, e, i, Om, w) overwritten from BSK orbital elements")
    pdf.bullet("Flowers re-placed along synced orbits via _ensure_reachable_flower_placement()")
    pdf.bullet("Task metadata assigned: task_id, task_description, deadline_step, status")

    pdf.subsubsection("Step (Per-Tick BSK Updates)")
    pdf.table(
        ["Phase", "What Happens"],
        [
            ["BSK Position Update", "bsk.step() advances Basilisk by 1s; new ECI -> grid coords -> bee.fx/fy/fz"],
            ["BSK Battery Sync", "bsk_battery_frac * battery_max[i] overwrites env battery"],
            ["Action Override", "Battery-dead bees forced to DONOTHING"],
            ["Proactive Groom", "If HARVEST but can't fit smallest flower -> auto-groom"],
            ["HARVEST Resolution", "Find closest reachable, non-expired, assigned/unassigned flower"],
            ["Task Expiration", "Flowers past deadline_step -> expired, penalty to owner"],
            ["Gossip Propagation", "Dead/orphan tasks relay hop-by-hop"],
        ],
        col_widths=[50, 140],
    )

    pdf.subsection("BSK vs RL Control Matrix")
    pdf.table(
        ["Aspect", "BSK Simulator", "RL Policy"],
        [
            ["Satellite Position", "Yes - Full N-body mechanics", "No - Overwritten by BSK"],
            ["Velocity", "Yes - Computed by BSK", "No - In state but not in obs"],
            ["Battery Drain", "Yes - SimplePowerSink", "No - Synced from BSK"],
            ["Solar Recharge", "Yes - SimpleSolarPanel", "No - Automatic in BSK"],
            ["Orbital Manoeuvring", "No - No thruster model", "No - Passive orbits"],
            ["Task Selection", "No", "Yes - HARVEST/GROOM/DONOTHING"],
            ["Task Assignment", "No", "Yes - Retask board + gossip"],
            ["Groom Timing", "No", "Yes - When to offload/recharge"],
            ["Task Expiration", "No", "Yes - Must harvest before deadline"],
        ],
        col_widths=[42, 74, 74],
    )

    pdf.info_box(
        "Key Insight",
        "The RL policy does NOT control orbital motion. Satellites follow their\n"
        "Basilisk-computed orbits passively. The policy's job is purely TASK\n"
        "SCHEDULING - given where each satellite IS (from BSK), choose whether\n"
        "to harvest, groom, or wait."
    )

    pdf.subsection("Element Sync Fix (0% to 54% Harvest)")
    pdf.body_text(
        "Initial BSK integration yielded 0% harvest because flowers were placed using "
        "default Keplerian elements (wide, eccentric orbits) while BSK positioned satellites "
        "in a tight Walker constellation. The fix syncs Bee Kepler elements FROM BSK orbital "
        "elements after initialization, then re-places flowers along the corrected orbits. "
        "Result: harvest rate jumped from 0% to 54% with the Keplerian-trained model."
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 6: TASK SYSTEM
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Task System & Metadata")

    pdf.body_text(
        "The task system was inspired by the BSU Autonomous Technologies Lab's "
        "satellite_constellation_scheduling repository. Each flower (task) carries rich "
        "metadata for realistic satellite task scheduling."
    )

    pdf.subsection("Task Descriptions")
    pdf.body_text("Tasks cycle through 5 description types based on flower index:")
    pdf.table(
        ["Index", "Task Description", "Analogy"],
        [
            ["0", "Capture imagery", "Earth observation / remote sensing"],
            ["1", "Relay communication", "Inter-satellite link relay"],
            ["2", "Data collection", "Sensor data aggregation"],
            ["3", "Sensor calibration", "Instrument maintenance"],
            ["4", "Harvest pollen", "General task completion"],
        ],
        col_widths=[20, 55, 115],
    )

    pdf.subsection("Task Lifecycle")
    pdf.code_block(
        "  CREATED (step 0)            task_id assigned, status='unassigned'\n"
        "       |                      deadline_step computed from priority\n"
        "       v\n"
        "  ASSIGNED                    status='assigned', bee gets flower in queue\n"
        "       |\n"
        "       +--> COMPLETED         status='completed' (harvested successfully)\n"
        "       |\n"
        "       +--> EXPIRED           status='expired' (past deadline_step)\n"
        "       |                      flower unassigned, -1.0 penalty to owner\n"
        "       |\n"
        "       +--> ORPHANED          owner bee died, tasks broadcast via gossip\n"
        "                              nearby bee can claim from retask board"
    )

    pdf.subsection("Deadline Computation")
    pdf.body_text(
        "Deadlines are priority-scaled: higher priority tasks have tighter deadlines. "
        "The formula is:"
    )
    pdf.code_block("deadline_step = int(max_steps * (0.3 + 0.5 * (1 - priority)))")
    pdf.body_text(
        "For max_steps=1200: a priority=1.0 task expires at step 360, while a priority=0.0 "
        "task expires at step 960. This forces the agent to prioritize high-value tasks."
    )

    pdf.subsection("Expiration Sweep")
    pdf.body_text(
        "Every step, the environment sweeps all flowers. Any unharvested flower whose "
        "current step exceeds its deadline_step is marked expired. The owning bee receives "
        "a -1.0 penalty, and the flower is excluded from all future harvest scans, retask "
        "board entries, and action availability calculations."
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 7: GOSSIP COMMUNICATION
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Gossip Communication Protocol")

    pdf.body_text(
        "Gossiper (gossiper.py, ~430 lines) implements a physics-based gossip relay protocol "
        "for decentralized task redistribution. When a satellite fails (battery=0), its "
        "orphan tasks are broadcast as GossipMessage objects that propagate hop-by-hop "
        "through the constellation."
    )

    pdf.subsection("GossipMessage Structure")
    pdf.table(
        ["Field", "Type", "Description"],
        [
            ["flower_id", "int", "Index of the orphaned flower/task"],
            ["priority", "float", "Task priority (pollen / capacity)"],
            ["origin_bee", "int", "Bee that died and released the task"],
            ["sender_bee", "int", "Last bee that forwarded the message"],
            ["x, y", "float", "Flower grid coordinates"],
            ["pollen", "float", "Pollen value of the task"],
            ["hops", "int", "Number of relay hops so far"],
            ["ttl", "int", "Time-to-live (max hops before discard)"],
            ["created_step", "int", "Step when message was created"],
            ["seen_by", "set[int]", "Set of bee IDs that have seen this message"],
        ],
        col_widths=[35, 25, 130],
    )

    pdf.subsection("Propagation Rules")
    pdf.bold_bullet("Range Limit", "Messages only reach bees within harvest_radius * 1.5 distance")
    pdf.bold_bullet("One Hop Per Step", "Each message advances one hop per environment step")
    pdf.bold_bullet("Claim vs Forward", "Receiving bee evaluates: reachability, capacity, deadline, workload. If suitable, claims task. If not, forwards to next unseen reachable bee.")
    pdf.bold_bullet("TTL Expiry", "Messages are discarded after TTL hops (prevents infinite forwarding)")
    pdf.bold_bullet("No Duplicates", "seen_by set prevents message loops")

    pdf.subsection("Retask Board Features (Per Slot)")
    pdf.table(
        ["Feature", "Description"],
        [
            ["x", "Normalized orphan flower x position"],
            ["y", "Normalized orphan flower y position"],
            ["pollen", "Pollen / max capacity"],
            ["can_fit", "1.0 if fits in bee's remaining capacity"],
            ["hops", "Gossip hop count / 10"],
        ],
        col_widths=[40, 150],
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 8: TRAINING PIPELINE
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Training Pipeline")

    pdf.subsection("PPO Algorithm")
    pdf.body_text(
        "Proximal Policy Optimization (PPO) is used for stable multi-agent training. "
        "Key advantages: trust-region updates prevent catastrophic policy changes, "
        "clipped surrogate objective ensures monotonic improvement, and entropy bonus "
        "maintains exploration."
    )

    pdf.subsubsection("Training Loop Pseudocode")
    pdf.code_block(
        "For update = 1 to 5000:\n"
        "  1. Collect rollout (768 steps in environment)\n"
        "     - For each step: observe, sample action, step env, store transition\n"
        "  2. Compute GAE advantages:\n"
        "     - TD error: d_t = r_t + gamma * V(s_{t+1}) - V(s_t)\n"
        "     - Advantage: A_t = sum(gamma*lambda)^k * d_{t+k}\n"
        "     - Returns: G_t = A_t + V(s_t)\n"
        "  3. PPO update (2 epochs, minibatch_size=128):\n"
        "     - Actor loss = -min(ratio*A, clip(ratio, 1-eps, 1+eps)*A) + entropy\n"
        "     - Critic loss = MSE(V(s), G)\n"
        "     - Backprop + Adam optimizer\n"
        "  4. Checkpoint if best mean reward\n"
        "  5. Early stop if harvest_rate >= 95% for 50 consecutive episodes"
    )

    pdf.subsection("Hyperparameters")
    pdf.table(
        ["Parameter", "Value", "Description"],
        [
            ["num_updates", "5000", "Maximum training iterations"],
            ["rollout_len", "768", "Steps per rollout before update"],
            ["gamma", "0.99", "Discount factor"],
            ["lambda (GAE)", "0.95", "Bias-variance tradeoff"],
            ["clip_ratio", "0.2", "PPO clipping (+/- 20%)"],
            ["value_coef", "0.5", "Critic loss weight"],
            ["entropy_coef", "0.02", "Exploration bonus strength"],
            ["lr_actor", "0.0003", "Actor learning rate (Adam)"],
            ["lr_critic", "0.0003", "Critic learning rate (Adam)"],
            ["num_epochs", "2", "PPO epochs per update"],
            ["minibatch_size", "128", "Mini-batch size"],
            ["patience", "1000", "Early stopping patience"],
            ["min_harvest_rate", "0.95", "Target harvest rate for early stop"],
        ],
        col_widths=[45, 30, 115],
    )

    pdf.subsection("Curriculum Learning")
    pdf.body_text(
        "5% of episodes use 'low-battery mode' with reduced capacity and increased drain. "
        "This forces the policy to learn proactive recharge, quick decision-making, and "
        "graceful degradation when teammates fail. Without this curriculum, agents learn "
        "lazy strategies that fail under stress."
    )

    pdf.subsection("Training Modes")
    pdf.table(
        ["Mode", "Command", "Output Dir"],
        [
            ["Standard (Keplerian)", "python train_orbital_v2.py --config config.yaml", "outputs/"],
            ["BSK Mode", "python train_orbital_v2.py --config config.yaml --bsk", "outputs_bsk/"],
            ["HRL Mode", "python train_orbital_v2.py --config config.yaml --hrl", "outputs_hrl/"],
            ["HRL + BSK", "python train_orbital_v2.py --config config.yaml --hrl --bsk", "outputs_hrl_bsk/"],
        ],
        col_widths=[40, 100, 50],
    )

    pdf.subsection("Model Checkpointing")
    pdf.body_text("Models are saved at multiple points during training:")
    pdf.bullet("best_actor.pt / best_critic.pt - Best mean reward achieved so far")
    pdf.bullet("upd{N}_actor.pt / upd{N}_critic.pt - Every 100 updates (for analysis)")
    pdf.bullet("final_actor.pt / final_critic.pt - At training completion")

    pdf.subsection("Resume Training")
    pdf.body_text("Training can be resumed from any checkpoint:")
    pdf.code_block(
        "python train_orbital_v2.py --config config.yaml \\\n"
        "    --resume outputs/upd400 --start-update 401"
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 9: BSK EVALUATOR & TELEMETRY
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("BSK Evaluator & Telemetry")

    pdf.body_text(
        "bsk_evaluator.py (~713 lines) runs the trained policy in BSK mode and produces "
        "telemetrybridge.json-format output for analysis and presentation."
    )

    pdf.subsection("Running the Evaluator")
    pdf.code_block(
        "# Set Basilisk path\n"
        "export PYTHONPATH=\"basilisk/dist3:$PYTHONPATH\"\n"
        "\n"
        "# Run BSK evaluation\n"
        "python bsk_evaluator.py \\\n"
        "    --model outputs/best_actor.pt \\\n"
        "    --num_sats 25 --num_tasks 50 \\\n"
        "    --steps 1200 --snapshot_interval 60 \\\n"
        "    --output bsk_telemetry_eval.json\n"
        "\n"
        "# Run without BSK (Keplerian fallback)\n"
        "python bsk_evaluator.py \\\n"
        "    --model outputs/best_actor.pt \\\n"
        "    --num_sats 25 --num_tasks 50 \\\n"
        "    --no_basilisk \\\n"
        "    --output keplerian_eval.json"
    )

    pdf.subsection("Telemetry Output Schema")
    pdf.body_text("Each snapshot includes per-satellite data:")
    pdf.table(
        ["Field", "Type", "Description"],
        [
            ["satellite_id", "str", "SAT-{id} identifier"],
            ["status", "str", "active / dead / charging"],
            ["orbit_type", "str", "NEO / MEO / GEO (based on altitude)"],
            ["position_eci", "dict", "x, y, z in metres (from BSK)"],
            ["velocity_eci", "dict", "vx, vy, vz in m/s"],
            ["position_rn", "dict", "range, latitude, longitude"],
            ["velocity_rn", "dict", "speed, heading"],
            ["assigned_tasks", "list", "Each with task_id, description, priority, status"],
            ["battery_level", "float", "0-100% charge"],
            ["fuel_mass", "float", "Fuel in kg (static 50 kg, no thruster)"],
            ["solar_panel_power", "float", "Solar output in watts"],
            ["communication_status", "str", "online / offline"],
            ["simulation_time", "float", "Basilisk sim time in seconds"],
        ],
        col_widths=[45, 25, 120],
    )

    pdf.subsection("Metrics Computed")
    pdf.table(
        ["Metric", "Description"],
        [
            ["harvest_rate", "Flowers harvested / total flowers"],
            ["expired_count", "Number of flowers that expired past deadline"],
            ["alive_count", "Satellites with battery > 0 at episode end"],
            ["mean_battery", "Average battery percentage across all satellites"],
            ["gossip_messages", "Total gossip messages sent during episode"],
            ["episode_end_state", "SUCCESS / PARTIAL / TIMEOUT"],
            ["total_steps", "Steps until episode termination"],
        ],
        col_widths=[50, 140],
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 10: CONFIGURATION REFERENCE
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Configuration Reference (config.yaml)")

    pdf.subsection("Environment Settings")
    pdf.table(
        ["Parameter", "Default", "Description"],
        [
            ["num_bees", "25", "Number of satellites"],
            ["num_flowers", "50", "Number of tasks"],
            ["grid_size", "75", "World size (grid_size x grid_size)"],
            ["max_steps", "1200", "Maximum episode length"],
            ["harvest_radius", "27.0", "Euclidean harvest distance"],
            ["reach_margin", "4.5", "Orbital reachability margin"],
            ["lambda_z", "0.5", "Z-axis weight in 3D distance"],
            ["orbit_scale", "0.8", "Orbital path scale factor"],
            ["time_window_min", "15", "Earliest window start step"],
            ["time_window_max", "120", "Latest window start step"],
            ["battery_min_steps", "450", "Minimum battery capacity"],
            ["battery_max_steps", "750", "Maximum battery capacity"],
            ["recharge_steps", "40", "Steps to fully recharge"],
            ["drain_per_step", "1.0", "Battery drain per step"],
            ["low_battery_chance", "0.05", "Probability of low-battery episode"],
            ["retask_board_size", "3", "Retask board slots per bee"],
            ["retask_timeout_steps", "40", "Steps before bee is 'silent'"],
            ["shaping_weight", "0.05", "Distance-based reward shaping weight"],
            ["anti_spam_pen", "-0.005", "Invalid harvest attempt penalty"],
            ["verbose", "false", "Detailed step logging"],
        ],
        col_widths=[48, 25, 117],
    )

    pdf.subsection("Training Settings")
    pdf.table(
        ["Parameter", "Default", "Description"],
        [
            ["hidden_dim", "256", "Network hidden layer size"],
            ["num_updates", "5000", "Max training iterations"],
            ["rollout_len", "768", "Steps per rollout"],
            ["gamma", "0.99", "Discount factor"],
            ["lam", "0.95", "GAE lambda"],
            ["clip_ratio", "0.2", "PPO clipping ratio"],
            ["value_coef", "0.5", "Value loss coefficient"],
            ["entropy_coef", "0.02", "Entropy bonus coefficient"],
            ["lr_actor", "0.0003", "Actor learning rate"],
            ["lr_critic", "0.0003", "Critic learning rate"],
            ["num_epochs", "2", "PPO epochs per update"],
            ["minibatch_size", "128", "Mini-batch size"],
            ["early_stopping", "true", "Enable early stopping"],
            ["patience", "1000", "Early stopping patience"],
            ["min_harvest_rate", "0.95", "Target harvest rate"],
            ["convergence_window", "50", "Performance check window"],
        ],
        col_widths=[48, 25, 117],
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 11: PERFORMANCE RESULTS
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Performance Results")

    pdf.subsection("Keplerian Mode (Training)")
    pdf.body_text("Trained with default Keplerian orbital dynamics, 25 bees, 50 flowers:")
    pdf.table(
        ["Metric", "Value"],
        [
            ["Best Harvest Rate", "93.9% (47/50 flowers)"],
            ["Training Updates", "~5000"],
            ["Convergence", "~500 updates to 80%, ~2000 to 90%+"],
            ["Device", "CUDA (GPU)"],
            ["Training Time", "~14 hours"],
        ],
        col_widths=[60, 130],
    )

    pdf.subsection("BSK Evaluation (Keplerian-Trained Model)")
    pdf.body_text(
        "The Keplerian-trained model was evaluated in BSK mode (Walker-delta constellation). "
        "Performance drops due to the geometry mismatch between training (spread Keplerian) "
        "and evaluation (tight Walker cluster)."
    )
    pdf.table(
        ["Metric", "Value"],
        [
            ["Harvest Rate", "54% (27/50 flowers)"],
            ["Expired Tasks", "23"],
            ["Alive Satellites", "25/25 (100%)"],
            ["Mean Battery", "79.5%"],
            ["Gossip Messages", "0 (no deaths)"],
            ["Episode End State", "PARTIAL (step 1199)"],
            ["Total Steps", "1199"],
        ],
        col_widths=[60, 130],
    )

    pdf.subsection("Performance Gap Analysis")
    pdf.body_text(
        "The 54% harvest rate in BSK mode (vs 93.9% in Keplerian) is explained by:"
    )
    pdf.bullet("Tight Walker geometry: BSK satellites cluster in 3 orbital planes vs spread Keplerian orbits")
    pdf.bullet("'No valid target' issue: Model over-selects HARVEST when no flower is within range")
    pdf.bullet("Task expirations: 23 flowers expired because satellites couldn't reach them before deadline")
    pdf.bullet("Zero-transfer gap: Model was never trained on BSK dynamics or Walker constellation geometry")
    pdf.ln(2)
    pdf.body_text(
        "Retraining directly on BSK mode (--bsk flag) is expected to close this gap "
        "significantly by learning Walker-specific scheduling behaviors."
    )

    pdf.subsection("BSK Training (In Progress)")
    pdf.body_text(
        "A new model is currently being trained with use_basilisk=True to learn directly "
        "on BSK Walker constellation dynamics. This training uses the same hyperparameters "
        "but with Basilisk providing high-fidelity orbital positions and battery state."
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 12: FILE REFERENCE
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("File Reference")

    pdf.table(
        ["File", "Lines", "Purpose"],
        [
            ["bees_env.py", "~2220", "PettingZoo multi-agent environment (core simulation)"],
            ["bee_state.py", "~570", "Bee + Flower classes with orbital mechanics + task metadata"],
            ["bee_policy.py", "~264", "Actor (transformer attention) + CentralizedCritic networks"],
            ["bsk_interface.py", "~420", "Basilisk wrapper - spacecraft creation, sim stepping"],
            ["gossiper.py", "~430", "Physics-based gossip relay for task redistribution"],
            ["train_orbital_v2.py", "~1250", "PPO training loop with BSK/HRL support"],
            ["bsk_evaluator.py", "~713", "BSK evaluation + telemetrybridge.json output"],
            ["telemetry_mapper.py", "~400", "Real telemetry JSON to simulation conversion"],
            ["config.yaml", "~90", "All training + environment hyperparameters"],
            ["bee_orbits_3d.py", "~500+", "3D matplotlib visualization with HUD"],
            ["train_utils.py", "~100+", "Config loading, model save/load utilities"],
            ["baseline_comparison.py", "~200", "Random, Greedy, Hungarian baselines"],
            ["test_model_telemetry.py", "~300", "Evaluate trained model on real scenarios"],
            ["mission_generator.py", "-", "Mission scenario generation"],
            ["mission_runner.py", "-", "Mission execution runner"],
        ],
        col_widths=[52, 20, 118],
    )

    pdf.subsection("Output Directories")
    pdf.table(
        ["Directory", "Contents"],
        [
            ["outputs/", "Keplerian-trained model checkpoints (best_actor.pt, etc.)"],
            ["outputs_bsk/", "BSK-trained model checkpoints"],
            ["outputs_hrl/", "HRL-trained model checkpoints (manager + worker)"],
            ["logs/", "Training logs and HUD snapshots"],
            ["basilisk/dist3/", "Basilisk compiled distribution (built from source)"],
        ],
        col_widths=[50, 140],
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 13: CLI COMMANDS
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("CLI Commands Reference")

    pdf.subsection("Training Commands")
    pdf.code_block(
        "# Standard training (Keplerian mode)\n"
        "python train_orbital_v2.py --config config.yaml --output outputs --seed 42\n"
        "\n"
        "# BSK training (Basilisk orbital dynamics)\n"
        "PYTHONPATH=\"basilisk/dist3:$PYTHONPATH\" \\\n"
        "  python train_orbital_v2.py --config config.yaml --output outputs --seed 42 --bsk\n"
        "\n"
        "# HRL training (Hierarchical RL)\n"
        "python train_orbital_v2.py --config config.yaml --output outputs --hrl\n"
        "\n"
        "# Resume from checkpoint\n"
        "python train_orbital_v2.py --config config.yaml --resume outputs/upd400 --start-update 401\n"
        "\n"
        "# Force CPU\n"
        "python train_orbital_v2.py --config config.yaml --cpu"
    )

    pdf.subsection("Evaluation Commands")
    pdf.code_block(
        "# BSK evaluation with trained model\n"
        "PYTHONPATH=\"basilisk/dist3:$PYTHONPATH\" \\\n"
        "  python bsk_evaluator.py \\\n"
        "    --model outputs/best_actor.pt \\\n"
        "    --num_sats 25 --num_tasks 50 \\\n"
        "    --steps 1200 --snapshot_interval 60 \\\n"
        "    --output bsk_telemetry_eval.json\n"
        "\n"
        "# Keplerian evaluation (no BSK)\n"
        "PYTHONPATH=\"basilisk/dist3:$PYTHONPATH\" \\\n"
        "  python bsk_evaluator.py \\\n"
        "    --model outputs/best_actor.pt \\\n"
        "    --no_basilisk \\\n"
        "    --output keplerian_eval.json"
    )

    pdf.subsection("Visualization Commands")
    pdf.code_block(
        "# 3D orbital visualization with trained policy\n"
        "python bee_orbits_3d.py --policy --model_tag best --stochastic\n"
        "\n"
        "# TensorBoard for training curves\n"
        "tensorboard --logdir outputs/"
    )

    pdf.subsection("CLI Flags Reference")
    pdf.table(
        ["Flag", "Type", "Default", "Description"],
        [
            ["--config", "str", "config.yaml", "Path to YAML config file"],
            ["--output", "str", "from config", "Output directory override"],
            ["--seed", "int", "42", "Random seed for reproducibility"],
            ["--cpu", "flag", "false", "Force CPU execution"],
            ["--bsk", "flag", "false", "Enable Basilisk orbital dynamics"],
            ["--hrl", "flag", "false", "Use Hierarchical RL (Manager+Worker)"],
            ["--resume", "str", "None", "Checkpoint directory to resume from"],
            ["--start-update", "int", "1", "Starting update number when resuming"],
            ["--manager-interval", "int", "10", "Steps between manager decisions (HRL)"],
        ],
        col_widths=[42, 18, 30, 100],
    )

    # ═══════════════════════════════════════════════════════════
    # SECTION 14: KNOWN ISSUES & FUTURE WORK
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Known Issues & Future Work")

    pdf.subsection("Known Issues")

    pdf.subsubsection("'No Valid Target' Harvest Attempts")
    pdf.body_text(
        "In BSK mode, the Keplerian-trained model frequently selects HARVEST when no flower "
        "is within harvest_radius. Root causes: (1) tight Walker geometry places fewer flowers "
        "within range, (2) expired flowers are invisible to the scan, (3) the model was trained "
        "on Keplerian orbits where flowers are more accessible."
    )
    pdf.body_text("Proposed solutions:")
    pdf.bullet("Action masking: override HARVEST -> DONOTHING when no valid target exists")
    pdf.bullet("Retrain on BSK: train directly with Basilisk dynamics (in progress)")
    pdf.bullet("Increase harvest_radius for BSK mode to account for Walker geometry")

    pdf.subsubsection("Velocity/Fuel Not in Observations")
    pdf.body_text(
        "BSK provides velocity and fuel mass per satellite, but these are not currently "
        "mapped into the observation space. Adding velocity could help the policy predict "
        "future positions and make better scheduling decisions."
    )

    pdf.subsubsection("Zero Gossip Messages in BSK Eval")
    pdf.body_text(
        "In BSK evaluation, 0 gossip messages were sent because no bees died (79.5% mean "
        "battery). The death-triggered gossip pathway is not exercised in the current BSK "
        "configuration. Low-battery episodes would trigger gossip."
    )

    pdf.subsection("Future Work")

    pdf.bold_bullet("BSK-Trained Model", "Complete BSK mode training and evaluate (currently in progress)")
    pdf.bold_bullet("Action Masking", "Implement hard action masking to prevent invalid HARVEST attempts")
    pdf.bold_bullet("Velocity Observations", "Map BSK velocity into observation space for better predictions")
    pdf.bold_bullet("Thruster Model", "Add Basilisk thruster module for orbital manoeuvring capability")
    pdf.bold_bullet("Multi-Orbit Types", "Support heterogeneous orbits (LEO + MEO + GEO mixed constellations)")
    pdf.bold_bullet("Communication Cost", "Add bandwidth constraints to gossip protocol")
    pdf.bold_bullet("Larger Swarms", "Scale to 50-100+ satellites and test coordination limits")
    pdf.bold_bullet("Real TLE Integration", "Load live satellite TLEs for real-world scenario testing")
    pdf.bold_bullet("Collision Avoidance", "Add proximity constraints and safe separation requirements")

    # ─── APPENDIX ───
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "Appendix A: Reward Structure Detail", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(20, 60, 120)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.table(
        ["Action / Event", "Reward", "Condition"],
        [
            ["Harvest success", "+5.0 + pollen/5", "Flower in range, not expired, assigned/unassigned"],
            ["Pollen bonus", "up to +10.0", "Higher pollen value = higher bonus"],
            ["Strategic groom", "+5.0 to +15.0", "Load >= 80% capacity, good timing"],
            ["Strategic recharge", "+8.0 to +13.5", "Battery < 20%, proactive recharge"],
            ["Proactive groom", "+reward", "Auto-groom when capacity can't fit flower"],
            ["Unnecessary groom", "-0.3", "Groom when load < 10%"],
            ["Invalid harvest", "-0.05 to -0.2", "No valid target within range"],
            ["Miss HARD window", "-50.0", "HARD window closed permanently"],
            ["Task expiration", "-1.0", "Flower passed deadline_step"],
            ["Time decay", "-0.01/step", "Encourages faster task completion"],
            ["DONOTHING w/ nearby flower", "small penalty", "Incentivizes action"],
        ],
        col_widths=[48, 42, 100],
    )

    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "Appendix B: Dependencies", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(20, 60, 120)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.code_block(
        "# requirements.txt\n"
        "torch>=2.0.0\n"
        "numpy>=1.21.0\n"
        "scipy>=1.9.0\n"
        "sgp4>=2.20\n"
        "matplotlib>=3.5.0\n"
        "Pillow>=9.0.0\n"
        "dash>=2.14.0\n"
        "dash-bootstrap-components>=1.5.0\n"
        "plotly>=5.18.0\n"
        "pettingzoo>=1.24.0\n"
        "gymnasium>=0.28.0\n"
        "tqdm>=4.64.0\n"
        "tensorboard>=2.12.0\n"
        "pyyaml>=6.0"
    )

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "Appendix C: Basilisk Setup", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(20, 60, 120)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.body_text("Basilisk is built from source and installed at basilisk/dist3/:")
    pdf.code_block(
        "# Build Basilisk from source (one-time)\n"
        "cd basilisk\n"
        "python conanfile.py\n"
        "# ... (follow BSK build instructions)\n"
        "\n"
        "# Set PYTHONPATH before running BSK-enabled commands\n"
        "export PYTHONPATH=\"/path/to/Bee_Fi/basilisk/dist3:$PYTHONPATH\"\n"
        "\n"
        "# Verify BSK is available\n"
        "python -c \"from Basilisk.utilities import SimulationBaseClass; print('BSK OK')\""
    )

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "Bee_Fi_System_Guide.pdf")
    pdf.output(out_path)
    print(f"PDF generated: {out_path}")
    print(f"  Pages: {pdf.page_no()}")
    return out_path


if __name__ == "__main__":
    build_guide()
