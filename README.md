# StructOpt (Experimental First-Order Optimizer)

StructOpt is an experimental first-order optimizer that uses a simple structural
signal based on gradient changes between steps.  
The goal of this prototype is to explore whether aspects of local loss surface
geometry can be partially recovered without Hessians or Hessian-vector products.

### Key Characteristics
- fully first-order  
- does not use second derivatives  
- computes a simple “landscape change” indicator  
- adaptively mixes two update regimes  

This repository contains:
- a minimal working demo on the Rosenbrock function  
- plots illustrating optimizer behavior  
- a concise technical description (without theoretical details)

⚠ **Note:** This is an early-stage research prototype.  
It is *not* a production optimizer and is intended only for experimentation.

---

## 🔧 Core Idea

On each step, the optimizer measures changes in the gradient:

`S_t = ||g_t – g_{t-1}|| / (||θ_t – θ_{t-1}|| + ε)`

S_t acts as a crude stiffness indicator:

- **low S_t** → flat region → accelerate  
- **high S_t** → sharp/unstable region → stabilize  

Mixing weight:

`α_t = sigmoid(a0 + a1 * ((S_t – S_ref) / (S_ref + ε)))`

Update step:

`Δθ = –η * ( α_t * d_t + (1 – α_t) * g_t )`

where d_t is a simple diagonal normalization of the gradient.

---

## 📁 Repository Contents

- `structopt_demo.py` – minimal optimizer demo  
- `loss_curve.png` – loss vs iterations  
- `signal_curve.png` – structural signal S(t)  
- `trajectory.png` – 2D trajectory on Rosenbrock  

---

## ⚠ Disclaimer
This project is provided solely for research and documentation purposes.  
It does not reveal any proprietary or sensitive theoretical details.
