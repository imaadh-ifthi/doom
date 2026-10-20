# System Limitations & Engineering Boundaries

## 1. Software Simulation vs. Physical FPGA Deployment
- All fixed-point operations, saturation clamping, and bitshift decay mechanisms are implemented via PyTorch software simulation.
- This code does not perform hardware synthesis, Verilog/VHDL code generation, place-and-route, or timing closure.
- Trainable parameter counts and INT16 memory estimates do not translate directly to FPGA LUT, DSP, or BRAM utilization.

## 2. ViZDoom Scenario Scope
- This work targets autonomous visual gameplay in the **ViZDoom BASIC** scenario.
- It does *not* claim general DOOM mastery across complex navigation maps, multi-room mazes, inventory management, or multi-weapon environments.

## 3. CNN vs. SNN Incommensurability
- Sujeevan’s CNN baseline operates on RGB $3 \times 60 \times 80$ frames, uses 3 buttons plus variable duration prediction, and was trained via imitation learning with privileged labels.
- SNN models operate on grayscale $1 \times 24 \times 32$ frames, use 6 compound actions with fixed 4-tic repeat, and learn via reinforcement learning without privileged labels.
- Results from the CNN and SNN tracks must not be directly compared as equivalent without explicitly noting these dimensional and methodological differences.

## 4. Surrogate Gradient Approximation
- Surrogate gradients (fast-sigmoid) are a software training heuristic enabling gradient-based backpropagation through non-differentiable threshold functions.
- Real neuromorphic hardware executes non-differentiable physical spike dynamics; gradients exist only on the host development machine during training.
