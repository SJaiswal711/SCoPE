# Parallel MCMC Sampler for Cosmological Parameter Inference

This code implements a **parallel Markov Chain Monte Carlo (MCMC)** sampler designed for cosmological parameter inference.

It uses **MPI for distributed computing** and incorporates several advanced techniques to accelerate convergence and likelihood evaluation:

* **Multiple chains** running concurrently with Gelman–Rubin convergence diagnostics.
* **Adaptive proposal covariance** using chain history to tune the proposal distribution.
* **Delayed Rejection (DR)** — a mathematically correct scheme that, upon rejection, attempts a second or third proposal with a narrower step size while maintaining detailed balance.
* **Dragging** — a fast-mode algorithm that efficiently samples nuisance (fast) parameters through multiple small steps while keeping cosmological (slow) parameters fixed.
* **Emulator** — Gaussian Process or polynomial models that predict CMB power spectra, avoiding expensive CAMB calls when prediction uncertainty is sufficiently low.
* **Fast-mode likelihood evaluation** that caches expensive likelihood components such as Commander and Lowlike during dragging steps.

This README provides an overview of the codebase and explains its main algorithms and execution flow.

---

## Table of Contents

* [General Structure](#general-structure)

  * [MPI Parallelisation](#mpi-parallelisation)
  * [Chains and Tasks](#chains-and-tasks)
  * [Master Loop](#master-loop)
  * [Slave Loop](#slave-loop)
* [Delayed Rejection](#delayed-rejection-dr)
* [Dragging Method](#dragging-method)
* [Emulator and Fast-Mode Likelihood](#emulator-and-fast-mode-likelihood)
* [Adaptive Proposal Covariance and Convergence](#adaptive-proposal-covariance-and-convergence)
* [Runtime Flags and Output](#runtime-flags-and-output)
* [Summary](#summary)

---

# General Structure

## MPI Parallelisation

The sampler uses **MPI** to distribute likelihood evaluations across multiple processes.

### Master Process (`rank 0`)

The master acts as the central orchestrator. It:

* Generates parameter proposals for each chain.
* Distributes proposals to slave processes.
* Collects likelihood results.
* Records accepted and rejected points.
* Updates chain files.
* Computes Gelman–Rubin convergence statistics.
* Adapts the proposal covariance matrix.

The actual Metropolis–Hastings decision is performed by the slaves, while the master manages the global chain state.

### Slave Processes (`rank >= 1`)

Slaves:

* Receive parameter vectors from the master.
* Evaluate the likelihood.
* Use either CAMB or the emulator.
* Return likelihood results to the chain middleman.

### Multiple Slaves per Chain

The constant

```c
SLAVEPARCHAIN
```

defines the number of slave processes assigned to each MCMC chain.

For each chain:

* The **first slave** acts as the **middleman**.
* The remaining slaves act as **workers**.

The middleman:

* Requests new tasks from the master.
* Evaluates the first proposal.
* Collects worker results.
* Performs the Delayed Rejection acceptance logic.
* Sends the final result to the master.

Workers evaluate additional proposals, primarily for the **Delayed Rejection** algorithm.

---

## Chains and Tasks

The number of chains is fixed at compile time:

```c
#define CHAINS 2
```

Each chain is represented as an array of `Task` structures.

A `Task` contains:

* `f[]` — a large array containing parameters, likelihoods, proposal information, and auxiliary data.
* `Multiplicity` — the number of times a point appears in the chain.
* `ReallyInvestigated` — the number of times the point has been proposed.

### Layout of `f[]`

The layout is defined by `setVariables()`.

Typical positions include:

| Location             | Contents                             |
| -------------------- | ------------------------------------ |
| `0 ... PARAMETERS-1` | Model parameters                     |
| `LOGLIKEPOS`         | Log-likelihood                       |
| `TAKEPOS`            | Acceptance decision                  |
| `PROBPOS`            | Proposal/stage information           |
| `RANDPOS`            | Random number information            |
| `STEP_POS`           | MCMC iteration counter               |
| `CURRENTSTATEPOS`    | Copy of the current parameter vector |

The exact layout allows the proposal, likelihood, acceptance state, and chain metadata to be passed between MPI processes using a single `Task`.

---

## Master Loop

The master enters a loop that continues until a termination condition is reached, such as the maximum chain length or proposal limit.

The basic structure is:

```text
Receive MPI message
        |
        +--> GIVEMETASK
        |       |
        |       +--> Generate proposals
        |       +--> Send proposals to slaves
        |
        +--> TAKERESULT
                |
                +--> Record result
                +--> Update chain
                +--> Adapt covariance
                +--> Check convergence
                +--> Check termination
```

### Handling `GIVEMETASK`

Only chain middlemen send `GIVEMETASK`.

When the master receives this message, it:

1. Determines the chain index.
2. Determines whether dragging should be used.
3. Generates one or more proposals.
4. Sends the proposals to the slaves assigned to that chain.

The number of proposals depends on whether Delayed Rejection is enabled.

* With DR enabled:

```text
SLAVEPARCHAIN proposals
```

* Without DR:

```text
1 proposal
```

Proposals are generated using:

```c
throwDice()
```

for the full parameter space, or:

```c
throwSlowDice()
```

when dragging is used.

The master also attaches:

* The current chain state.
* Step number.
* Random-number information.
* Dragging metadata.

Each proposal is then sent to its designated slave:

```c
MPI_Send(..., TAKETASK)
```

---

## Handling `TAKERESULT`

When the master receives a final result from a middleman, it:

1. Writes the evaluated point to the `investigated` file.
2. Reads the Metropolis–Hastings decision from `TAKEPOS`.
3. Updates the chain multiplicity.
4. If accepted:

   * Appends the new point to the chain.
   * Moves the chain to the new state.
5. Periodically updates the proposal covariance.
6. Computes convergence statistics.
7. Checks termination conditions.

When dragging is enabled, the master also broadcasts the fast-parameter covariance to the slaves.

---

## Slave Loop

Each slave runs continuously until it receives a stop signal.

The general structure is:

```text
Check for TAG_STOP
        |
        v
Middleman requests task
        |
        v
Receive task
        |
        +--> Evaluate proposal
        |
        +--> Worker sends result to middleman
        |
        +--> Middleman performs MH/DR logic
                |
                v
            Send final result to master
```

Each slave:

1. Checks for `TAG_STOP`.
2. If it is a middleman, sends `GIVEMETASK`.
3. Waits for:

   * `TAKETASK`, or
   * `TAG_UPDATE_COV`.
4. Evaluates the proposal using:

```c
evaluate_proposal()
```

The proposal may be evaluated using:

* CAMB,
* the emulator,
* dragging,
* or fast-mode likelihood evaluation.

### Workers

Workers simply evaluate their assigned proposal and return the result to their chain middleman:

```c
MPI_Send(..., TAKERESULT)
```

### Middlemen

The middleman:

1. Keeps its own result as **stage 0**.
2. Performs standard Metropolis–Hastings acceptance.
3. If rejected, receives additional worker results.
4. Applies Delayed Rejection.
5. Sends the final accepted or rejected result to the master.

If a proposal is accepted and CAMB was used rather than the emulator, the middleman updates the emulator with the true CMB spectra.

---

# Delayed Rejection (DR)

## What Is Delayed Rejection?

In standard Metropolis–Hastings sampling:

1. Propose a new point.
2. Accept or reject it.
3. If rejected, remain at the current point.

**Delayed Rejection**, following Tierney & Mira, allows the sampler to attempt another proposal after rejection.

The second proposal is typically drawn from a narrower distribution.

For example:

```text
Stage 0: Full proposal step
        |
        | Rejected
        v
Stage 1: Smaller proposal step
        |
        | Rejected
        v
Stage 2: Even smaller proposal step
```

The acceptance probability must be corrected to account for the rejected earlier proposals.

This correction ensures that **detailed balance is preserved**.

---

## Configuration

Delayed Rejection can be enabled or disabled at runtime:

```text
-dr
-no-dr
```

When DR is disabled:

```text
SLAVEPARCHAIN = 1
```

Only one proposal is evaluated per chain.

---

## Proposal Scales

The proposal scale decreases with the DR stage.

The scale is controlled by:

```text
drfactor = 0.7
```

For stage `i`, the effective scale is:

```text
pow(drfactor, i)
```

Therefore:

| Stage | Scale   |
| ----- | ------- |
| 0     | `1.0`   |
| 1     | `0.7`   |
| 2     | `0.49`  |
| 3     | `0.343` |

Each later proposal is smaller and therefore more likely to fall near the current high-probability region.

---

## Stage 0

The first proposal uses the normal proposal distribution.

The acceptance probability is:

```text
alpha0 = min(1, exp(logL_prop - logL_cur))
```

If:

```text
u < alpha0
```

the proposal is accepted.

---

## Later Stages

If stage 0 is rejected, the middleman receives results from worker processes.

For stage `i`:

1. A smaller proposal has already been generated by the master.
2. The corresponding worker evaluates its likelihood.
3. The middleman computes the DR correction.
4. The proposal is accepted or rejected.

The correction contains:

* Likelihood ratios.
* Proposal density ratios.
* The probability of rejecting previous proposals.

A representative calculation is:

```text
alpha12 = min(1, exp(0.5 * (newss - loglike_cur)))
alpha32 = min(1, exp(0.5 * (newss - logL_worker)))

l2 = exp(0.5 * (logL_worker - loglike_cur))

q1 = exp(-0.5 * (rand_norm - rand_old))

alpha13 = l2 * q1 * (1 - alpha32) / (1 - alpha12)
```

The final decision is:

```text
if u < alpha13:
    accept proposal
else:
    continue to the next stage
```

If no stage is accepted, the chain remains at the current point.

---

## Why Use Delayed Rejection?

Delayed Rejection improves efficiency when:

* The initial proposal is too large.
* The local posterior structure is narrow.
* A rejected proposal does not necessarily imply that all nearby proposals should be rejected.

Instead of immediately wasting the iteration, the sampler attempts smaller moves.

Because the acceptance probability is formally corrected, the target posterior distribution remains unchanged.

---

# Dragging Method

## Motivation

Cosmological parameter spaces often contain two types of parameters:

### Slow Parameters

These are parameters that significantly affect the CMB power spectra.

Examples include:

* `Omega_m_h2`
* `Omega_b_h2`
* `h`
* `tau`
* `n_s`
* `A_s`

Changing these parameters may require expensive CAMB calculations.

### Fast Parameters

Fast parameters typically include nuisance parameters such as:

* Calibration factors.
* Foreground amplitudes.
* Instrumental nuisance parameters.

These parameters can often be changed without recomputing the expensive cosmological quantities.

---

## Parameter Classification

Parameters are classified using:

```c
classify_parameters()
```

The code separates parameters into:

```text
Slow parameters
Fast parameters
```

This classification allows different sampling strategies for each group.

---

## Fast Covariance Matrix

Once sufficient chain history is available:

```text
BEGINCOVUPDATE
```

the master:

1. Computes the full covariance matrix.
2. Extracts the fast-parameter covariance submatrix.
3. Computes its eigenvalues and eigenvectors.
4. Broadcasts the result to all slaves.

The update uses:

```text
TAG_UPDATE_COV
```

Each slave stores:

```text
fast_eval
fast_evec
```

and sets:

```text
fast_cov_ready = 1
```

This covariance describes efficient directions for moving through the fast-parameter subspace.

---

## Dragging Algorithm

Dragging is enabled when a proposal contains:

```text
DRAGPOS = 1
```

Suppose:

* `A` is the current state.
* `B` is the proposed state.

The slow parameters are gradually moved from `A` to `B`.

For:

```text
d = 1 ... N_DRAG
```

where the default is:

```text
N_DRAG = 3
```

the interpolation weight is:

```text
w = d / N_DRAG
```

The slow parameters are interpolated between the two endpoints.

Meanwhile, the fast parameters receive small Gaussian perturbations generated using the fast covariance matrix.

Conceptually:

```text
A
|
|  Small fast-parameter adjustments
v
Intermediate state 1
|
|  Small fast-parameter adjustments
v
Intermediate state 2
|
v
B
```

---

## Dragging Steps

The algorithm:

1. Computes the likelihood for the current state `A`.
2. Computes the likelihood for the proposed slow state `B`.
3. Caches expensive likelihood components.
4. Moves gradually through the slow-parameter space.
5. Performs small fast-parameter moves at each intermediate point.
6. Applies Metropolis–Hastings acceptance to each segment.
7. Accumulates the required acceptance information.
8. Performs a final acceptance decision.

If accepted:

* The slow parameters correspond to the proposed state.
* The fast parameters correspond to the final dragging trajectory.

---

## Why Dragging Is Efficient

The key advantage is that expensive likelihood components can be reused.

Rather than performing a complete likelihood evaluation at every intermediate step, the sampler caches the components that remain unchanged.

This is particularly effective when:

* There are many nuisance parameters.
* Slow parameters are computationally expensive.
* Fast parameters require substantial exploration.

---

# Emulator and Fast-Mode Likelihood

## Emulator

The emulator predicts CMB power spectra:

* `Cl_TT`
* `Cl_TE`
* `Cl_EE`
* `Cl_BB`

from the slow cosmological parameters.

Its purpose is to avoid expensive CAMB calculations when a sufficiently accurate approximation is available.

Supported approaches include:

* Gaussian Process emulation.
* Polynomial emulation.

Each slave initializes its own emulator instance:

```c
g_emulator
```

when:

```text
USE_EMULATOR
```

is enabled.

---

## Online Training

The emulator is trained during the MCMC run.

Whenever a proposal is:

1. Accepted, and
2. Evaluated using CAMB rather than the emulator,

the sampler stores:

* The slow parameter vector.
* The true CMB spectra.

This is performed through:

```text
emulator_update()
```

When the training buffer reaches approximately:

```text
200 points
```

the emulator is trained:

```text
emulator_train()
```

---

## Emulator Prediction

Inside:

```c
evaluate_proposal()
```

the sampler checks whether the emulator is sufficiently reliable.

If:

```text
prediction uncertainty < FALLBACK_THRESHOLD
```

then:

```text
use_emu = 1
```

and the emulator spectra are used.

Otherwise:

```text
use_emu = 0
```

and CAMB is called.

The prediction uncertainty is also displayed in the progress information.

This produces a hybrid strategy:

```text
Low uncertainty
      |
      v
Use Emulator
```

```text
High uncertainty
      |
      v
Use CAMB
```

---

# Fast-Mode Likelihood

The likelihood contains three major components:

```text
Total likelihood
       |
       +--> CAMspec
       |
       +--> Commander
       |
       +--> Lowlike
```

Commander and Lowlike are expensive components associated with the slow cosmological state.

---

## Caching

During dragging, the code caches:

```text
Commander + Lowlike
```

as:

```text
cached_other_chi2
```

These values can then be reused across intermediate dragging steps.

In fast mode:

```text
fast_mode = 1
```

the cached contribution is reused while only the necessary remaining likelihood component is recomputed.

This reduces the computational cost of dragging substantially.

Instead of repeatedly evaluating every expensive component:

```text
Full likelihood
Full likelihood
Full likelihood
Full likelihood
```

the code effectively performs:

```text
Expensive component: computed once
Fast component:      recomputed as needed
```

For approximately `N_DRAG + 1` intermediate evaluations, this can provide a substantial speedup.

---

# Adaptive Proposal Covariance and Convergence

## Adaptive Covariance

The master maintains proposal covariance information for the chains.

After sufficient chain history is available:

```text
BEGINCOVUPDATE
```

the covariance matrix is periodically recomputed.

Updates occur approximately every:

```text
UPDATE_TIME = 5
```

accepted steps.

The covariance is estimated from recent chain history, weighted using point multiplicity.

The sampler then:

1. Computes the covariance matrix.
2. Performs eigenvalue/eigenvector decomposition using GSL.
3. Updates the proposal distribution.

This allows the sampler to gradually learn the geometry of the posterior distribution.

---

## Adaptive Proposal Scale

A global scale factor:

```text
EntireFactor
```

controls the overall proposal size.

It is adjusted according to the acceptance behavior:

```text
Too many acceptances
        |
        v
Increase proposal size
```

```text
Too many rejections
        |
        v
Decrease proposal size
```

This helps maintain an efficient proposal scale.

---

## Gelman–Rubin Convergence

The code periodically computes the Gelman–Rubin statistic:

```text
R
```

across multiple chains.

The diagnostic compares:

* Variance within individual chains.
* Variance between chains.

Convergence is assumed when all parameters satisfy:

```text
R < RBREAK
```

with:

```text
RBREAK = 1.2
```

and the chains contain at least:

```text
MIN_SIZE_FOR_FREEZE_IN
```

points.

---

## Freeze-In

Once convergence criteria are satisfied:

```c
ADAPTIVE = 0
```

Proposal adaptation stops.

The proposal distribution is then frozen.

The sampler:

* Stops modifying the covariance matrix.
* Uses a fixed proposal distribution.
* Creates new chain files for the post-burn-in phase.

This is important because continuously adapting the proposal can complicate the theoretical properties of the Markov chain.

---

# Runtime Flags and Output

## Runtime Flags

The program supports the following runtime options:

| Flag       | Description                  |
| ---------- | ---------------------------- |
| `-restart` | Restart a previous run       |
| `-o <dir>` | Specify the output directory |
| `-dr`      | Enable Delayed Rejection     |
| `-no-dr`   | Disable Delayed Rejection    |
| `-emu`     | Enable the emulator          |
| `-no-emu`  | Disable the emulator         |
| `-drag`    | Enable dragging              |
| `-no-drag` | Disable dragging             |

---

## Output Files

The sampler writes several files to the specified output directory.

### Accepted Chains

```text
montecarlo_chain_N.d
```

Contains:

* Accepted parameter points.
* Log-likelihood values.
* Multiplicity information.

---

### Chain Head

```text
head_N.d
```

Contains the latest proposal or current state associated with each chain.

---

### Investigated Points

```text
investigated_N.d
```

Contains every evaluated proposal, including rejected points.

This is useful for:

* Debugging.
* Proposal diagnostics.
* Acceptance-rate analysis.

---

### Progress Information

```text
progress.txt
```

Contains information about the progress of the run.

This may include:

* Chain progress.
* Likelihood information.
* Emulator usage.
* Emulator uncertainty.

---

### Gelman–Rubin Diagnostics

```text
gelmanRubin.txt
```

Contains the convergence statistic history.

---

### Covariance Matrices

```text
covMatrix.txt
```

Contains proposal covariance matrices produced during adaptive updates.

---

## Timing

The total elapsed execution time is printed at the end of the run.

---

# Execution Architecture

The overall MPI workflow can be summarized as:

```text
                         ┌─────────────┐
                         │   MASTER    │
                         │   Rank 0    │
                         └──────┬──────┘
                                │
                    Generate proposals
                                │
               ┌────────────────┼────────────────┐
               │                │                │
               ▼                ▼                ▼
          ┌─────────┐      ┌─────────┐      ┌─────────┐
          │ Chain 0 │      │ Chain 1 │      │   ...   │
          │Middleman│      │Middleman│      │Middleman│
          └────┬────┘      └────┬────┘      └────┬────┘
               │                │
               │                │
        ┌──────┴──────┐  ┌──────┴──────┐
        ▼             ▼  ▼             ▼
     Worker 1      Worker 2       Workers
        │             │
        └──────┬──────┘
               │
         Likelihood results
               │
               ▼
          Middleman
               │
       MH / Delayed Rejection
               │
               ▼
            MASTER
```

---

# Key Features

## Delayed Rejection

Attempts smaller proposals after rejection while preserving detailed balance.

```text
Large proposal
      |
      ├── Accepted → Move
      |
      └── Rejected
              |
              ▼
        Smaller proposal
              |
              ├── Accepted → Move
              |
              └── Rejected
                      |
                      ▼
                Even smaller proposal
```

---

## Dragging

Efficiently explores fast nuisance parameters while gradually moving through the slow cosmological parameter space.

---

## Emulator

Avoids expensive CAMB calculations when the predicted CMB spectra are sufficiently reliable.

---

## Fast-Mode Likelihood

Caches expensive likelihood components during dragging.

---

## Adaptive Proposal

Learns the posterior covariance structure from the chain history.

---

## Gelman–Rubin Diagnostics

Uses multiple chains to monitor convergence and determines when proposal adaptation can be frozen.

---

# Summary

This code is a **parallel MCMC sampler tailored for cosmological parameter inference**.

Its primary features are:

* **MPI parallelisation** for distributed likelihood evaluation.
* **Multiple concurrent chains**.
* **Gelman–Rubin convergence diagnostics**.
* **Adaptive proposal covariance**.
* **Delayed Rejection** for improved mixing without biasing the posterior.
* **Dragging** for efficient exploration of nuisance parameters.
* **Gaussian Process or polynomial emulation** of CMB power spectra.
* **Fast-mode likelihood caching** to reduce computational cost.
* **Automatic proposal freeze-in** after convergence.

The MPI architecture allows the sampler to scale across multiple cores. Each chain can use multiple slave processes, with one **middleman** coordinating worker results and performing the Metropolis–Hastings and Delayed Rejection acceptance logic.

Together, **Delayed Rejection, dragging, emulation, adaptive proposals, and cached likelihood evaluations** provide a computationally efficient framework for exploring high-dimensional cosmological parameter spaces.
