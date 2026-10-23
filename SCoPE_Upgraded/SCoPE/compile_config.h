#ifndef _COMPILE_CONFIG_H_
#define _COMPILE_CONFIG_H_

/* ============================================================
   compile_config.h

   Single header for EVERY numeric constant in the build.

   SECTION 1 — COMPILE-TIME CONSTANTS
       Array sizes, struct field sizes, MPI topology. These are
       resolved by the compiler and CANNOT be set from the ini.
       To change one: edit here, then  ./compile.sh

   SECTION 2 — RUNTIME DEFAULTS
       Default values for sampler / emulator / I/O tunables.
       These can be overridden per-run from param_reader.ini
       (the parser still recognises the keys) or from CLI flags
       where noted. If left unset, the defaults below are used.

   The ini parser has NO branches for SECTION 1 names; adding
   one to param_reader.ini would make load_config() treat it as
   a parameter declaration and exit(1).
   ============================================================ */


/* ============================================================
   SECTION 1 — COMPILE-TIME CONSTANTS
   ============================================================ */

/* ---- Parameter storage (Config in param_config_reader.h) ---- */
#define MAX_PARAMS             350      /* was 200 — supports Plik 2015 */
#define MAX_LINE_LENGTH        512
#define MAX_PARAM_NAME_LENGTH  100
#define MAX_LIKELIHOODS         10

/* ---- MCMC task array (mcmc_scope.c) ---- */
#define MAX_TASKARRAY_SIZE     350

/* ---- Fast-parameter covariance buffers (mcmc_scope.c) ---- */
#define MAX_FAST               120
#define MAX_FAST_PACKED        (1 + MAX_FAST + MAX_FAST*MAX_FAST)

/* ---- Parameter-index array capacity (mcmc_scope.c)
        MUST be >= the number of SAMPLED parameters (PARAMETERS)
        at runtime, i.e. the total of all `... sample flat/gaussian ...`
        lines in param_reader.ini across every active profile.

        Current ceilings to fit:
          Profile 1 (2013 CAMspec) : 20
          Profile 2 (2018 full)    : 53
          Profile 3 (2018 lite)    :  7
          Profile 4 (2015 full)    : 100
          Profile 5 (2015 lite)    :  7

        NOPARAM = MAX_PARAMS (350) covers all of the above with
        room to spare; costs ~17 KB of static storage. ---- */
#define NOPARAM                MAX_PARAMS

/* ---- Covariance-file parser (mcmc_scope.c) ---- */
#define MAX_FILE_PARAMS        200

/* ---- MPI topology — must match `mpirun -np` ---- */
#define CHAINS                  2
#define SLAVEPARCHAIN           1

/* ---- Emulator rolling-average window (mcmc_scope.c) ---- */
#define EMU_BIAS_WINDOW        50

/* ---- MPI message tags (master <-> slave wire protocol) ----
        Must be unique per communicator and lie in
        [0, MPI_TAG_UB] (typically >= 32767).
        1..4 are handshake tags; 400+ are side-channel tags
        that are polled via MPI_Iprobe/MPI_Probe so they can
        arrive asynchronously while a normal task is in flight.
        Do not renumber without checking every MPI_Send /
        MPI_Recv / MPI_Probe site in mcmc_scope.c. ---- */
#define GIVEMETASK             1
#define TAKERESULT             2
#define TAKETASK               3
#define TAG_STOP               4
#define TAG_UPDATE_COV         400
#define TAG_EMU_QUARANTINE     401

/* ============================================================
   SECTION 2 — RUNTIME DEFAULTS
   (ini may override; otherwise these values are used)
   ============================================================ */

/* ---- Sampler ---- */
#define DEF_MAX_CHAIN_LENGTH       50000
#define DEF_BEGIN_COV_UPDATE       120
#define DEF_UPDATE_TIME            5
#define DEF_LOW_STEP_BOUND         3
#define DEF_HIGH_STEP_BOUND        5
#define DEF_DECREASE_STEP          0.9
#define DEF_INCREASE_STEP          (1.0 / 0.9)
#define DEF_BEGIN_DRAGGING         150
#define DEF_N_DRAG                 3
#define DEF_RMIN_POINTS            120
#define DEF_RBREAK                 1.2
#define DEF_MIN_SIZE_FOR_FREEZE_IN 500
#define DEF_MAX_PREVIOUS_POINTS    5000
#define DEF_INITIAL_DRFACTOR       0.7

/* ---- Dragging (CLI: -drag / -no-drag) ---- */
#define DEF_USE_DRAGGING           1

/* ---- Emulator (CLI: -emu / -no-emu, -train-fallback-only) ---- */
#define DEF_USE_EMULATOR           1
#define DEF_EMULATOR_TYPE          1        /* 0 = GP, 1 = polynomial */
#define DEF_POLY_DEGREE            2
#define DEF_FALLBACK_THRESHOLD     0.2
#define DEF_TRAIN_ON_FALLBACK_ONLY 0
#define DEF_EMU_BIAS_MIN_N         30
#define DEF_N_RECOVERY             100
#define DEF_MAX_EMU_RESCUES        20
#define DEF_EMU_BIAS_MAX           5.0
#define DEF_FREE_REJECT_DCHI2_MAX  10.0
#define DEF_FREE_REJECT_RATIO      2

/* ---- I/O (CLI: -print / -no-print, -restart, -o outdir, -load-cov) ---- */
#define DEF_VERBOSE                1
#define DEF_RESTART                0
#define DEF_OUTDIR                 "."
#define DEF_LOAD_COV_PATH          ""       /* empty = no preload */

#endif /* _COMPILE_CONFIG_H_ */
