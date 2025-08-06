"""
HNX-M_v4 (Public Research Release)
----------------------------------
This file contains the public top-level model definition for HNX-M_v4,
a dual-strand, entropy-gated neural architecture for memory-efficient sequence learning.

Core implementations for imported components in `core/` are proprietary and omitted.
They are covered under patent protection.

For research discussion and evaluation purposes only.

Non‑commercial Academic Research License
"""

# === Public model file (core components omitted) ===

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.metadata import init_meta_logs, update_meta_logs, finalize_meta_logs
from core.gates import EntropyGate, compute_entropy
from core.scan import StrandScanContrastProcessor
from core.rung_logic import compute_state_delta, compute_rung_signal
from core.strand import (
    StrandProcessor, LearnableDecayMemory,
    BackwardScaler, project_with_torsion_split, flip_time
)

class HNX_M_V4(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_memory_slots):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, 2 * hidden_dim)
        self.fusion_weight = nn.Parameter(torch.ones(3))  # [fwd, bwd, mem]
        self.forward_strand = StrandProcessor(dim=hidden_dim, direction="forward", kernel_size=2)
        self.backward_strand = StrandScanContrastProcessor(dim=hidden_dim)

        self.memory = LearnableDecayMemory(num_memory_slots, hidden_dim)  # Updated memory
        self.mem_gate = nn.Linear(hidden_dim, 1)
        self.entropy_gate = EntropyGate(hidden_dim)
        self.backward_scaler = BackwardScaler()
        self.slot_proj = nn.Linear(hidden_dim, num_memory_slots)
        self.slot_proj_bwd = nn.Linear(hidden_dim, num_memory_slots)
        self.output_proj = nn.Linear(hidden_dim, output_dim)  # not 3 * hidden_dim

        self.dt_bias_fwd = nn.Parameter(torch.zeros(self.hidden_dim))
        self.dt_bias_bwd = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, x):
        meta_logs = init_meta_logs()
        B, T, _ = x.shape

        # === Torsion Phase + Gate Split ===
        x = project_with_torsion_split(x, self.input_proj, self.hidden_dim, self.dt_bias_fwd)

        # === Forward Strand Encoding ===
        fwd_out = self.forward_strand(x)

        # === Compute Rung Signal ===
        h_prev = torch.roll(fwd_out, shifts=1, dims=1)
        delta = compute_state_delta(fwd_out, h_prev)
        rung_signal = compute_rung_signal(delta)

        # === Gated Memory Write ===
        entropy = compute_entropy(fwd_out)
        B, T, D = fwd_out.shape
        
        # === Forward Slot Weights + Write ===
        slot_logits = self.slot_proj(fwd_out)             # [B, T, N]
        slot_weights = torch.softmax(slot_logits, dim=-1) # [B, T, N]

        # === Backward Slot Weights + Read ===
        slot_weights_bwd = torch.softmax(self.slot_proj_bwd(fwd_out.detach()), dim=-1)  # detach to avoid reuse gradients
        mem_out = self.memory.read_all(slot_weights_bwd)  # [B, T, D]

        # === Apply per-token memory gate ===
        mem_gate = torch.sigmoid(self.mem_gate(fwd_out))  # [B, T, 1]
        mem_out = mem_out * mem_gate                     # gated memory injection

        # === Meta Logs ===
        slot_sparsity = (F.softmax(fwd_out, dim=-1) > 0.1).sum(dim=-1).float()
        update_meta_logs(meta_logs, {
            "trigger_values": rung_signal.squeeze(-1),
            "entropy_values": entropy,
            "slot_sparsity": slot_sparsity,
            "rung_trigger_map": rung_signal.squeeze(-1),
            "memory_weights": torch.softmax(fwd_out, dim=-1)
        }, batch_size=B)

        # === Backward Strand ===
        entropy_val = entropy.mean(dim=-1)
        gate_val = self.backward_scaler(entropy_val)

        bwd_in = flip_time(fwd_out.detach())                   # reverse time
        bwd_scanned = self.backward_strand(bwd_in)
        bwd_out = flip_time(bwd_scanned)                       # flip back to normal
        bwd_out = gate_val.unsqueeze(-1) * bwd_out             # entropy gate

        meta_logs["backward_gate_log"].append(gate_val.detach().cpu())
        meta_logs["memory_per_token"] = mem_out.detach().cpu()          # [B, T, D]
        meta_logs["memory_weights_full"] = slot_weights.detach().cpu()  # [B, T, N]
        meta_logs["memory_gate"] = mem_gate.detach().cpu()

        # === Output Projection ===
        combined = torch.stack([fwd_out, bwd_out, mem_out], dim=0)  # [3, B, T, D]
        weighted = (self.fusion_weight[:, None, None, None] * combined).sum(dim=0)
        return self.output_proj(weighted), finalize_meta_logs(meta_logs)
