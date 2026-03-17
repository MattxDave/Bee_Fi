# 🐝 Bee Swarm RL System - Architecture Documentation

## System Overview

```mermaid
flowchart TB
    subgraph System["🐝 BEE SWARM MULTI-AGENT RL SYSTEM"]
        direction LR
        
        subgraph Training["📚 TRAINING LOOP<br/>train_orbital_v2.py"]
            T1[PPO Algorithm]
            T2[16 Parallel Envs]
            T3[Rollout Buffer]
        end
        
        subgraph Env["🌍 ENVIRONMENT<br/>BeeForagingEnv"]
            E1[Multi-Agent PG]
            E2[Orbital Mechanics]
            E3[Battery/Recharge]
            E4[Retasking]
        end
        
        subgraph Viz["📊 VISUALIZATION<br/>bee_orbits_3d.py"]
            V1[3D Orbital View]
            V2[Real-time HUD]
            V3[Event Logging]
            V4[Communication Viz]
        end
        
        Training <--> Env
        Env <--> Viz
    end
    
    style System fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style Training fill:#2d2d44,stroke:#00ff88,color:#fff
    style Env fill:#2d2d44,stroke:#ffaa00,color:#fff
    style Viz fill:#2d2d44,stroke:#ff6b6b,color:#fff
```

---

## 1. Neural Network Architecture

### Actor Network (Per-Agent Policy)

```mermaid
flowchart LR
    subgraph Inputs["📥 INPUTS (Per-Bee Observation)"]
        direction TB
        POS["Position (3D)<br/>[x, y, z] / 30"]
        STAT["Status (2)<br/>[load, capacity]"]
        FLOW["Flowers (N×12)<br/>Per flower features"]
        STEP["Step Count (1)<br/>[step/max_step]"]
        CONS["Consensus (N)<br/>[last_actions]"]
        RETASK["Retask Board (M×5)<br/>Per slot features"]
        AVAIL["Action Avail (3)<br/>[can_harv, groom, idle]"]
    end
    
    subgraph Encoders["🔧 FEATURE ENCODERS"]
        direction TB
        E1["Linear(3→64)<br/>LayerNorm + ReLU"]
        E2["Linear(2→32)<br/>LayerNorm + ReLU"]
        E3["Linear(12→128)<br/>LayerNorm + ReLU"]
        E4["Linear(1→32)<br/>LayerNorm + ReLU"]
        E5["Linear(N→64)<br/>LayerNorm + ReLU"]
        E6["Linear(5M→128)<br/>LayerNorm + ReLU"]
        E7["Linear(3→32)<br/>LayerNorm + ReLU"]
    end
    
    subgraph Attention["🎯 FLOWER ATTENTION"]
        direction TB
        TRANS["Transformer Encoder<br/>d_model=128, heads=4<br/>1 layer"]
        POOL["Mean Pooling<br/>(B, 128)"]
    end
    
    subgraph Trunk["🧠 TRUNK NETWORK"]
        direction TB
        CONCAT["Concatenate<br/>480 dims"]
        L1["Linear(480→256)<br/>LayerNorm + ReLU"]
        L2["Linear(256→256)<br/>LayerNorm + ReLU"]
    end
    
    subgraph Heads["📤 OUTPUT HEADS"]
        direction TB
        POLICY["Policy Head<br/>Linear(256→3)"]
        CLAIM["Claim Head<br/>Linear(256→M+1)"]
        
        ACTS["🎬 Actions<br/>DONOTHING<br/>HARVEST<br/>GROOM"]
        SLOTS["📋 Retask Slots<br/>Slot 0..M-1<br/>or No-Claim"]
    end
    
    POS --> E1
    STAT --> E2
    FLOW --> E3
    STEP --> E4
    CONS --> E5
    RETASK --> E6
    AVAIL --> E7
    
    E3 --> TRANS --> POOL
    
    E1 --> CONCAT
    E2 --> CONCAT
    POOL --> CONCAT
    E4 --> CONCAT
    E5 --> CONCAT
    E6 --> CONCAT
    E7 --> CONCAT
    
    CONCAT --> L1 --> L2
    
    L2 --> POLICY --> ACTS
    L2 --> CLAIM --> SLOTS
    
    style Inputs fill:#1e3a5f,stroke:#00d4ff,color:#fff
    style Encoders fill:#2d4a3e,stroke:#00ff88,color:#fff
    style Attention fill:#4a3a2d,stroke:#ffaa00,color:#fff
    style Trunk fill:#3a2d4a,stroke:#aa66ff,color:#fff
    style Heads fill:#4a2d2d,stroke:#ff6b6b,color:#fff
```

### Centralized Critic Network

```mermaid
flowchart LR
    subgraph GlobalState["📥 GLOBAL STATE INPUT"]
        direction TB
        BEES["All Bees Concatenated<br/>Bee 0: [x,y,z, load, cap, bat]<br/>Bee 1: [x,y,z, load, cap, bat]<br/>...<br/>Bee N: [x,y,z, load, cap, bat]"]
        FLOWERS["All Flowers Concatenated<br/>Flower 0: [x,y, harvested, ...]<br/>Flower 1: [x,y, harvested, ...]<br/>..."]
    end
    
    subgraph CriticTrunk["🧠 CRITIC TRUNK"]
        direction TB
        C1["Linear(608→512)<br/>LayerNorm + ReLU"]
        C2["Linear(512→512)<br/>LayerNorm + ReLU"]
    end
    
    subgraph ValueOut["📤 VALUE OUTPUT"]
        VHEAD["Value Head<br/>Linear(512→1)"]
        VALUE["V(s) scalar<br/>(state value)"]
    end
    
    BEES --> C1
    FLOWERS --> C1
    C1 --> C2
    C2 --> VHEAD --> VALUE
    
    style GlobalState fill:#1e3a5f,stroke:#00d4ff,color:#fff
    style CriticTrunk fill:#3a2d4a,stroke:#aa66ff,color:#fff
    style ValueOut fill:#2d4a3e,stroke:#00ff88,color:#fff
```

---

## 2. Per-Bee Retask Board & Communication

```mermaid
flowchart TB
    subgraph Boards["🐝 DECENTRALIZED RETASK BOARDS (Each Bee Has Its Own)"]
        direction LR
        
        subgraph B0["BEE 0"]
            RB0["retask_board<br/>━━━━━━━━━━<br/>📋 Task 1<br/>📋 Task 4"]
        end
        
        subgraph B1["BEE 1"]
            RB1["retask_board<br/>━━━━━━━━━━<br/>(empty)"]
        end
        
        subgraph B2["BEE 2"]
            RB2["retask_board<br/>━━━━━━━━━━<br/>📋 Task 3"]
        end
        
        subgraph B3["BEE 3"]
            RB3["retask_board<br/>━━━━━━━━━━<br/>(empty)"]
        end
        
        subgraph B4["BEE 4"]
            RB4["retask_board<br/>━━━━━━━━━━<br/>📋 Task 2"]
        end
    end
    
    subgraph TaskStruct["📦 TASK STRUCTURE"]
        TS["flower_id: 5<br/>source_bee: 2<br/>hops: 1<br/>received_step: 42<br/>pollen: 7.5"]
    end
    
    style Boards fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style B0 fill:#2d4a3e,stroke:#00ff88,color:#fff
    style B1 fill:#3a3a3a,stroke:#888,color:#fff
    style B2 fill:#2d4a3e,stroke:#00ff88,color:#fff
    style B3 fill:#3a3a3a,stroke:#888,color:#fff
    style B4 fill:#2d4a3e,stroke:#00ff88,color:#fff
    style TaskStruct fill:#4a3a2d,stroke:#ffaa00,color:#fff
```

### Communication Flow (When Bee Dies)

```mermaid
sequenceDiagram
    participant B0 as 🐝 Bee 0<br/>(Battery = 0)
    participant SYS as 🌍 Environment
    participant B2 as 🐝 Bee 2<br/>(Nearest Active)
    
    Note over B0: 💀 BATTERY DEPLETED!<br/>Has flowers: [F5, F7, F9]
    
    B0->>SYS: I'm dying! Find nearest bee
    SYS->>SYS: Calculate 3D distances<br/>to all active bees
    SYS-->>B0: Nearest = Bee 2 (dist: 5.3)
    
    rect rgb(40, 60, 40)
        Note over B0,B2: 📡 BROADCAST TASKS
        B0->>B2: Task F5 {src:0, hops:1}
        B0->>B2: Task F7 {src:0, hops:1}
        B0->>B2: Task F9 {src:0, hops:1}
    end
    
    Note over B0: assigned = []<br/>(cleared)
    Note over B2: retask_board now has<br/>3 new tasks!
```

### Task Propagation Logic (Each Step)

```mermaid
flowchart TB
    START["🔄 _propagate_retask_board()<br/>Called each step"]
    
    FILTER["🧹 Filter Stale Tasks<br/>(received > 100 steps ago)"]
    
    FOREACH["📋 For Each Task in Board"]
    
    CHECK{"🎯 Can I reach<br/>this flower?"}
    
    subgraph ClaimPath["✅ CAN REACH"]
        CLAIM["CLAIM IT!<br/>━━━━━━━━━━<br/>flower.assigned_bee = me<br/>Add to assigned_flowers[]"]
    end
    
    subgraph HandoffPath["❌ CANNOT REACH"]
        MARK["Mark for Handoff<br/>awaiting_handoff = True"]
        FIND["Find nearest bee<br/>within comm range<br/>(2× harvest_radius)"]
        PASS["Pass task to that bee<br/>task.hops += 1<br/>other.retask_board.append()"]
    end
    
    START --> FILTER --> FOREACH --> CHECK
    CHECK -->|Yes| CLAIM
    CHECK -->|No| MARK --> FIND --> PASS
    
    style START fill:#1e3a5f,stroke:#00d4ff,color:#fff
    style FILTER fill:#4a3a2d,stroke:#ffaa00,color:#fff
    style CHECK fill:#3a2d4a,stroke:#aa66ff,color:#fff
    style ClaimPath fill:#2d4a3e,stroke:#00ff88,color:#fff
    style HandoffPath fill:#4a2d2d,stroke:#ff6b6b,color:#fff
```

---

## 3. Environment Architecture

```mermaid
flowchart TB
    subgraph Config["⚙️ CONFIGURATION"]
        direction LR
        C1["num_bees: 5<br/>num_flowers: 12<br/>grid_size: 30<br/>max_steps: 800"]
        C2["harvest_radius: 12.0<br/>reach_margin: 2.0<br/>lambda_z: 0.5<br/>orbit_scale: 0.8"]
        C3["battery_min: 300<br/>battery_max: 500<br/>recharge_time: 30<br/>retask_board_size: 2"]
    end
    
    subgraph BeeObj["🐝 BEE OBJECT (bee_state.py)"]
        direction TB
        
        subgraph Orbital["🛰️ ORBITAL PARAMETERS"]
            O1["a: semi-major axis<br/>e: eccentricity<br/>i: inclination"]
            O2["Ω: RAAN<br/>ω: arg of periapsis<br/>ν: true anomaly<br/>T: orbital period"]
        end
        
        subgraph Status["📊 STATUS"]
            S1["fx, fy, fz: position<br/>load: current pollen<br/>capacity: max pollen"]
            S2["battery: energy left<br/>mode: IDLE/HARV/GROOM<br/>truncated: done flag<br/>assigned_flowers[]"]
        end
        
        subgraph Comm["📡 COMMUNICATION"]
            CM["retask_board[]<br/>last_broadcast_to<br/>last_received_from<br/>awaiting_handoff"]
        end
    end
    
    subgraph FlowerObj["🌸 FLOWER OBJECT"]
        direction TB
        F1["id: flower index<br/>x, y: grid position<br/>pollen: amount (1-10)<br/>priority: importance"]
        F2["harvested: bool<br/>assigned_bee: int/None<br/>busy_by: int/None"]
        F3["window_type: NONE/SOFT/HARD<br/>window_start, window_end"]
    end
    
    subgraph PosUpdate["📐 POSITION UPDATE"]
        PU["Kepler equation + rotation matrix<br/>fx, fy, fz = R₃(Ω) × R₁(i) × R₃(ω) × [a·cos(ν), b·sin(ν), 0]"]
    end
    
    Config --> BeeObj
    Config --> FlowerObj
    BeeObj --> PosUpdate
    
    style Config fill:#1e3a5f,stroke:#00d4ff,color:#fff
    style BeeObj fill:#2d4a3e,stroke:#00ff88,color:#fff
    style Orbital fill:#3a4a3e,stroke:#88ff88,color:#fff
    style Status fill:#3a4a3e,stroke:#88ff88,color:#fff
    style Comm fill:#3a4a3e,stroke:#88ff88,color:#fff
    style FlowerObj fill:#4a3a2d,stroke:#ffaa00,color:#fff
    style PosUpdate fill:#3a2d4a,stroke:#aa66ff,color:#fff
```

### Action & Observation Space

```mermaid
flowchart LR
    subgraph Actions["🎬 ACTION SPACE"]
        direction TB
        A0["0️⃣ DONOTHING<br/>Continue orbiting"]
        A1["1️⃣ HARVEST<br/>Collect pollen if in range"]
        A2["2️⃣ GROOM<br/>Deposit pollen, reset load"]
        AC["➕ CLAIM HEAD<br/>Select retask slot 0..M-1<br/>or M = no-claim"]
    end
    
    subgraph Observations["👁️ OBSERVATION SPACE (Per Bee)"]
        direction TB
        OB1["position (3)<br/>[x, y, z] / grid_size"]
        OB2["status (2)<br/>[load/capacity, capacity_norm]"]
        OB3["flowers (N × 12)<br/>x,y,pollen,flags,dist,time"]
        OB4["action_availability (3)<br/>[can_harvest, can_groom, can_idle]"]
        OB5["step_count (1)<br/>current_step / max_steps"]
        OB6["consensus (num_bees)<br/>last actions of all bees"]
        OB7["retask_board (M × 5)<br/>x,y,priority,reachable,assigned"]
    end
    
    style Actions fill:#4a2d2d,stroke:#ff6b6b,color:#fff
    style Observations fill:#1e3a5f,stroke:#00d4ff,color:#fff
```

---

## 4. Training Loop (PPO)

```mermaid
flowchart TB
    subgraph VecEnvs["🌐 VECTORIZED ENVIRONMENTS (16 parallel)"]
        direction LR
        E0["Env 0<br/>5 bees<br/>12 flowers"]
        E1["Env 1<br/>5 bees<br/>12 flowers"]
        E2["Env 2<br/>5 bees<br/>12 flowers"]
        DOTS["..."]
        E15["Env 15<br/>5 bees<br/>12 flowers"]
    end
    
    COLLECT["📦 Batched Collection<br/>(rollout_len steps)"]
    
    subgraph Buffer["💾 ROLLOUT BUFFER"]
        direction TB
        BUF["For each step × env × bee:<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>obs[t] | action[t] | log_prob[t]<br/>reward[t] | value[t] | done[t]<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Shape: (rollout_len × num_envs × num_bees, ...)"]
        GAE["📊 Compute Advantages<br/>GAE(λ=0.95, γ=0.99)<br/>Normalize: (adv - mean) / (std + ε)"]
    end
    
    subgraph PPOUpdate["🔄 PPO UPDATE (4 epochs)"]
        direction TB
        SHUFFLE["🔀 Shuffle Minibatches"]
        
        subgraph ActorLoss["📉 ACTOR LOSS"]
            AL["ratio = exp(log_prob_new - log_prob_old)<br/>L_clip = min(ratio × Â, clip(ratio, 1-ε, 1+ε) × Â)<br/>L_actor = -mean(L_clip) + entropy_bonus × H(π)"]
        end
        
        subgraph CriticLoss["📉 CRITIC LOSS"]
            CL["L_critic = MSE(V(s), returns)<br/>(with gradient clipping)"]
        end
        
        BACK["⬅️ Backprop + Adam optimizer step"]
    end
    
    VecEnvs --> COLLECT --> Buffer
    Buffer --> PPOUpdate
    GAE --> SHUFFLE
    SHUFFLE --> ActorLoss
    SHUFFLE --> CriticLoss
    ActorLoss --> BACK
    CriticLoss --> BACK
    
    style VecEnvs fill:#1e3a5f,stroke:#00d4ff,color:#fff
    style Buffer fill:#2d4a3e,stroke:#00ff88,color:#fff
    style PPOUpdate fill:#4a3a2d,stroke:#ffaa00,color:#fff
    style ActorLoss fill:#4a2d4a,stroke:#ff66aa,color:#fff
    style CriticLoss fill:#4a2d4a,stroke:#ff66aa,color:#fff
```

---

## 5. Reward Structure

```mermaid
flowchart LR
    subgraph Rewards["🎁 REWARD FUNCTION"]
        direction TB
        
        subgraph Harvest["🌸 HARVEST"]
            H1["✅ Success: +5.0 + pollen/5"]
            H2["🔄 Load full → groom: +5.0"]
            H3["❌ Out of range: -1.0"]
        end
        
        subgraph Groom["✨ GROOM"]
            G1["✅ Strategic (load ≥ 80%): +5.0 + 10×load_ratio"]
            G2["⚠️ Unnecessary (load < 10%): -0.5"]
            G3["⚠️ Premature (load < 80%): -0.3"]
        end
        
        subgraph DoNothing["💤 DONOTHING"]
            D1["⚠️ At high capacity: small penalty"]
            D2["✅ No flowers in range: 0"]
        end
        
        subgraph Episode["🏁 EPISODE END"]
            EP1["🎉 All flowers harvested: SUCCESS"]
            EP2["⏰ Max steps reached: TIMEOUT"]
            EP3["💀 All bees truncated: FAILED"]
        end
    end
    
    style Rewards fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style Harvest fill:#2d4a3e,stroke:#00ff88,color:#fff
    style Groom fill:#4a3a2d,stroke:#ffaa00,color:#fff
    style DoNothing fill:#3a3a4a,stroke:#888,color:#fff
    style Episode fill:#4a2d4a,stroke:#aa66ff,color:#fff
```

---

## 6. Visualization Pipeline

```mermaid
flowchart TB
    subgraph VizLayout["📊 bee_orbits_3d.py VISUALIZATION"]
        direction LR
        
        subgraph MainView["🌐 3D ORBITAL VIEW"]
            MV1["Orbits + Bees + Flowers"]
            MV2["🐝 Bee positions with trails"]
            MV3["🌸 Flower positions"]
            MV4["━━ Task assignment lines"]
            MV5["┅┅ Retask claim lines"]
        end
        
        subgraph Panels["📋 HUD PANELS"]
            direction TB
            
            subgraph P1["🐝 BEE STATUS"]
                BS["ID | Battery | Load | Status<br/>━━━━━━━━━━━━━━━━━━━━━━━<br/>0  | ████░░ | 5/10 | HARVEST<br/>1  | ██████ | 0/10 | IDLE<br/>2  | ░░░░░░ | 0/10 | 💀 DEAD"]
            end
            
            subgraph P2["🎯 MISSION STATUS"]
                MS["Progress: [████████░░] 8/12<br/>━━━━━━━━━━━━━━━━━━━━━━━<br/>📋 RETASK QUEUE:<br/>#0: Flower 5 (prio: 7.5)"]
            end
            
            subgraph P3["📡 COMMUNICATION LOG"]
                CL["[42] 💀 Bee 2 DIED!<br/>[42] 📋 Flower 5 orphaned<br/>[43] 🔄 F5: Bee 2 → Bee 0<br/>[48] ✅ Flower 5 harvested"]
            end
        end
    end
    
    style VizLayout fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style MainView fill:#2d2d44,stroke:#00ff88,color:#fff
    style Panels fill:#2d2d44,stroke:#ffaa00,color:#fff
    style P1 fill:#3a3a4a,stroke:#00d4ff,color:#fff
    style P2 fill:#3a3a4a,stroke:#00ff88,color:#fff
    style P3 fill:#3a3a4a,stroke:#ff6b6b,color:#fff
```

---

## 7. Key Files Reference

```mermaid
flowchart TB
    subgraph Files["📁 PROJECT STRUCTURE"]
        direction TB
        
        subgraph Core["🔧 CORE FILES"]
            F1["bee_state.py<br/>━━━━━━━━━━━━━<br/>Bee class with<br/>orbital mechanics +<br/>retask_board"]
            F2["bees_env.py<br/>━━━━━━━━━━━━━<br/>PettingZoo multi-agent<br/>environment"]
            F3["bee_policy.py<br/>━━━━━━━━━━━━━<br/>Actor (attention) +<br/>CentralizedCritic"]
        end
        
        subgraph Training["📚 TRAINING"]
            F4["train_orbital_v2.py<br/>━━━━━━━━━━━━━<br/>PPO training with<br/>parallel envs"]
        end
        
        subgraph Viz["📊 VISUALIZATION"]
            F5["bee_orbits_3d.py<br/>━━━━━━━━━━━━━<br/>3D visualization<br/>with HUD"]
        end
        
        subgraph Models["💾 SAVED MODELS"]
            F6["best/best_actor.pt<br/>Trained actor weights"]
            F7["best/best_critic.pt<br/>Trained critic weights"]
        end
    end
    
    F1 --> F2
    F3 --> F4
    F2 --> F4
    F4 --> Models
    F2 --> F5
    F3 --> F5
    Models --> F5
    
    style Files fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style Core fill:#2d4a3e,stroke:#00ff88,color:#fff
    style Training fill:#4a3a2d,stroke:#ffaa00,color:#fff
    style Viz fill:#3a2d4a,stroke:#aa66ff,color:#fff
    style Models fill:#4a2d2d,stroke:#ff6b6b,color:#fff
```

---

## 8. Hyperparameters

```mermaid
flowchart LR
    subgraph Hyperparams["⚙️ TRAINING HYPERPARAMETERS"]
        direction TB
        
        subgraph PPO["📊 PPO"]
            P1["γ (gamma): 0.99"]
            P2["λ (GAE lambda): 0.95"]
            P3["ε (clip): 0.2"]
            P4["entropy_coef: 0.01"]
            P5["learning_rate: 3e-4"]
            P6["num_epochs: 4"]
            P7["num_envs: 16"]
            P8["rollout_len: 256"]
        end
        
        subgraph Network["🧠 NETWORK"]
            N1["hidden_dim: 256"]
            N2["attention_heads: 4"]
            N3["transformer_layers: 1"]
            N4["flower_embed_dim: 128"]
        end
        
        subgraph Environment["🌍 ENVIRONMENT"]
            E1["num_bees: 5"]
            E2["num_flowers: 12"]
            E3["grid_size: 30"]
            E4["max_steps: 800"]
            E5["harvest_radius: 12"]
            E6["retask_board_size: 2"]
        end
    end
    
    style Hyperparams fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style PPO fill:#2d4a3e,stroke:#00ff88,color:#fff
    style Network fill:#3a2d4a,stroke:#aa66ff,color:#fff
    style Environment fill:#4a3a2d,stroke:#ffaa00,color:#fff
```

---

## 9. Full System Data Flow

```mermaid
flowchart TB
    subgraph DataFlow["🔄 COMPLETE SYSTEM DATA FLOW"]
        direction TB
        
        ENV["🌍 Environment<br/>BeeForagingEnv"]
        
        subgraph PerBee["👁️ Per-Bee Observation"]
            OBS["position + status + flowers<br/>+ consensus + retask_board<br/>+ action_availability"]
        end
        
        ACTOR["🎭 Actor Network<br/>(Attention-based)"]
        
        subgraph Actions["🎬 Actions"]
            ACT["action: 0/1/2<br/>claim: 0..M"]
        end
        
        subgraph Rewards["🎁 Rewards"]
            REW["harvest: +5.0<br/>groom: +5.0<br/>penalty: -1.0"]
        end
        
        subgraph GlobalState["🌐 Global State"]
            GS["All bees + All flowers<br/>concatenated"]
        end
        
        CRITIC["📈 Centralized Critic"]
        
        VALUE["V(s) Value Estimate"]
        
        subgraph PPOLoss["📉 PPO Losses"]
            LOSS["Actor: Clipped Surrogate<br/>Critic: MSE(V, returns)"]
        end
        
        OPTIM["⚡ Adam Optimizer"]
        
        ENV --> PerBee --> ACTOR --> Actions --> ENV
        ENV --> Rewards
        ENV --> GlobalState --> CRITIC --> VALUE
        Rewards --> PPOLoss
        VALUE --> PPOLoss
        Actions --> PPOLoss
        PPOLoss --> OPTIM
        OPTIM -->|Update weights| ACTOR
        OPTIM -->|Update weights| CRITIC
    end
    
    style DataFlow fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style ENV fill:#2d4a3e,stroke:#00ff88,color:#fff
    style ACTOR fill:#3a2d4a,stroke:#aa66ff,color:#fff
    style CRITIC fill:#3a2d4a,stroke:#aa66ff,color:#fff
    style PPOLoss fill:#4a2d2d,stroke:#ff6b6b,color:#fff
```
