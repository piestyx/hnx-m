# TensorBoard Logs — Minigrid PPO Test 1

This folder contains event files for the **first Minigrid PPO benchmark** run with `HNX-M_v4`.

---

## Setup Summary

**Environment**
- ID: `MiniGrid-GoToObject-8x8-N2-v0`
- Wrapper: `ImgObsWrapper` (image‑only observations)
- Parallel environments: 4 (`make_vec_env`)

**PPO Configuration**
- Policy: `CnnPolicy`
- Features Extractor: `HNXMFeatures` (proprietary, part of HNX‑M core)
- Feature dimension: `128`
- Total timesteps: `100,000`
- Framework: Stable‑Baselines3
- TensorBoard log dir: `ppo_minigrid_tensorboard/`

**Hardware**
- GPU: RX 7900 XTX - AMD RDNA3 (ROCm)
- CPU: Ryzen 7 7800X3D - 8 threads for environment rollout

---

## Viewing the Logs

To view interactively in TensorBoard:

```bash
tensorboard --logdir logs/minigrid_ppo_test1
````

Then open:
[http://localhost:6006](http://localhost:6006) in your browser.

You’ll be able to explore:

* Training reward curves
* Episode length
* Loss metrics
* PPO‑specific stats

---

## Notes

* This is the **first** standardised benchmark for HNX‑M in a Minigrid environment.
* Further Minigrid and Minihack benchmarks will be added as testing progresses.
* The `HNXMFeatures` extractor is proprietary and not included in this public release.