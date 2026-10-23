#ifndef _PARAM_CONFIG_H_
#define _PARAM_CONFIG_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "compile_config.h"   /* MAX_* caps + DEF_* runtime defaults */

typedef enum { PARAM_CLASS_COSMO, PARAM_CLASS_NUISANCE } ParameterClass;
typedef enum { PARAM_SPEED_SLOW, PARAM_SPEED_FAST } ParameterSpeed;
typedef enum { PARAM_FIXED, PARAM_SAMPLED } ParameterState;
typedef enum { USAGE_UNKNOWN = 0, USAGE_CAMB, USAGE_NUISANCE, USAGE_DERIVED } ParamUsage;
typedef enum { PRIOR_FLAT, PRIOR_GAUSSIAN } PriorType;

typedef struct {
    char name[MAX_PARAM_NAME_LENGTH];
    ParameterClass class_type;
    ParameterSpeed speed;
    ParameterState state;
    PriorType prior_type;
    double fixed_value;
    double lower_bound;
    double upper_bound;
    double proposal_sigma;
    double gaussian_mean;
    double gaussian_sigma;
    int has_bounds;
    int has_sigma;
    int is_estimated;

    /* Layer C: optional `default=X` for nuisance/SN fallback */
    int    has_default;
    double default_value;

    ParamUsage usage;
    int target_index;
} ParameterConfig;

typedef struct {
    char likelihood_paths[MAX_LIKELIHOODS][256];
    int  lh_nuisance_exist[MAX_LIKELIHOODS];
    int  lh_nuisance_sample[MAX_LIKELIHOODS];
    int  likelihood_count;

    /* Layer A/B/C config flags */
    int  use_sn;
    int  sample_nuisance;
    char camb_highl_template[256];
    char sn_data_path[256];
    char sn_cov_path[256];

    /* ============================================================
       RUNTIME TUNABLES — settable from param.ini
       ============================================================ */

    /* Sampler */
    int    max_chain_length;
    int    begin_cov_update;
    int    update_time;
    int    low_step_bound;
    int    high_step_bound;
    double decrease_step;
    double increase_step;
    int    begin_dragging;
    int    n_drag;
    int    rmin_points;
    double rbreak;
    int    min_size_for_freeze_in;
    int    max_previous_points;
    double initial_drfactor;

    /* Dragging */
    int    use_dragging;

    /* Emulator */
    int    use_emulator;
    int    emulator_type;
    int    poly_degree;
    double fallback_threshold;
    int    train_on_fallback_only;
    int    emu_bias_min_n;
    int    n_recovery;
    int    max_emu_rescues;
    double emu_bias_max;
    double free_reject_dchi2_max;
    int    free_reject_ratio;

    /* I/O */
    int    verbose;
    int    restart;
    char   outdir[512];
    char   load_cov_path[512];

    ParameterConfig params[MAX_PARAMS];
    int param_count;

    int num_cosmo_sampled;
    int num_nuis_sampled;
    int num_sampled_total;
    int num_cosmo_fixed;
    int num_nuis_fixed;
    int num_slow_sampled;
    int num_fast_sampled;
} Config;

extern Config global_config;

static char* trim(char* str) {
    char* end;
    while(isspace((unsigned char)*str)) str++;
    if(*str == 0) return str;
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

static int parse_parameter_line(char* line, ParameterConfig* param) {
    char* equals_pos = strchr(line, '=');
    if(!equals_pos) return 0;

    *equals_pos = '\0';
    char* name = trim(line);
    char* value_str = trim(equals_pos + 1);

    char clean_name[MAX_PARAM_NAME_LENGTH];
    int j = 0;
    for(int i = 0; name[i] != '\0' && j < MAX_PARAM_NAME_LENGTH - 1; i++) {
        if(name[i] != '$' && name[i] != '{' && name[i] != '}' &&
           name[i] != '^' && name[i] != '\\' && !isspace(name[i])) {
            clean_name[j++] = name[i];
        }
    }
    clean_name[j] = '\0';

    strncpy(param->name, clean_name, MAX_PARAM_NAME_LENGTH - 1);
    param->name[MAX_PARAM_NAME_LENGTH - 1] = '\0';

    char class_str[32], speed_str[32], state_str[32];
    int parsed = sscanf(value_str, "%31s %31s %31s", class_str, speed_str, state_str);
    if(parsed < 3) {
        fprintf(stderr, "[FATAL ERROR] Incomplete declaration for parameter '%s'. Must specify class, speed, and state.\n", param->name);
        exit(1);
    }

    if(strcmp(class_str, "cosmo") == 0) param->class_type = PARAM_CLASS_COSMO;
    else if(strcmp(class_str, "nuis") == 0) param->class_type = PARAM_CLASS_NUISANCE;
    else { fprintf(stderr, "[FATAL ERROR] Unknown parameter class '%s' for '%s'.\n", class_str, param->name); exit(1); }

    if(strcmp(speed_str, "slow") == 0) param->speed = PARAM_SPEED_SLOW;
    else if(strcmp(speed_str, "fast") == 0) param->speed = PARAM_SPEED_FAST;
    else { fprintf(stderr, "[FATAL ERROR] Unknown speed '%s' for '%s'.\n", speed_str, param->name); exit(1); }

    if(strcmp(state_str, "sample") == 0) {
        param->state = PARAM_SAMPLED;
        param->is_estimated = 1;

        char prior_str[32];
        char* token = strstr(value_str, "sample") + 6;
        int parsed_prior = sscanf(token, "%31s", prior_str);

        if (parsed_prior > 0 && strcmp(prior_str, "flat") == 0) {
            param->prior_type = PRIOR_FLAT;
            token = strstr(token, "flat") + 4;
            int parsed_nums = sscanf(token, "%lf %lf %lf", &param->lower_bound, &param->upper_bound, &param->proposal_sigma);
            if(parsed_nums != 3) {
                fprintf(stderr, "[FATAL ERROR] Flat parameter '%s' requires: lower upper proposal.\n", param->name);
                exit(1);
            }
        } else if (parsed_prior > 0 && strcmp(prior_str, "gaussian") == 0) {
            param->prior_type = PRIOR_GAUSSIAN;
            token = strstr(token, "gaussian") + 8;
            int parsed_nums = sscanf(token, "%lf %lf %lf %lf %lf", &param->lower_bound, &param->upper_bound, &param->gaussian_mean, &param->gaussian_sigma, &param->proposal_sigma);
            if(parsed_nums != 5) {
                fprintf(stderr, "[FATAL ERROR] Gaussian parameter '%s' requires: lower upper mean sigma proposal.\n", param->name);
                exit(1);
            }
        } else {
            fprintf(stderr, "[FATAL ERROR] Sampled parameter '%s' must specify 'flat' or 'gaussian'.\n", param->name);
            exit(1);
        }

        param->has_bounds = 1;
        param->has_sigma = 1;
        param->fixed_value = 0.0;
    }
    else if(strcmp(state_str, "fixed") == 0) {
        param->state = PARAM_FIXED;
        param->is_estimated = 0;

        char* token = strstr(value_str, "fixed") + 5;
        double dummy;
        int parsed_nums = sscanf(token, "%lf %lf", &param->fixed_value, &dummy);
        if(parsed_nums != 1) {
            fprintf(stderr, "[FATAL ERROR] Fixed parameter '%s' requires exactly ONE value.\n", param->name);
            exit(1);
        }
        param->has_bounds = 0;
        param->has_sigma = 0;
        param->lower_bound = NAN;
        param->upper_bound = NAN;
        param->proposal_sigma = 0.0;
    }
    else {
        fprintf(stderr, "[FATAL ERROR] Unknown state '%s' for '%s'.\n", state_str, param->name);
        exit(1);
    }

    /* Layer C: parse optional `default=X.XX` keyword */
    param->has_default   = 0;
    param->default_value = 0.0;
    if (param->state == PARAM_SAMPLED) {
        char *dpos = strstr(value_str, "default=");
        if (dpos) {
            double dv = 0.0;
            if (sscanf(dpos + 8, "%lf", &dv) == 1) {
                param->has_default   = 1;
                param->default_value = dv;

                if (dv < param->lower_bound || dv > param->upper_bound) {
                    fprintf(stderr,
                        "[WARNING] '%s': default=%.4f is outside bounds [%.4f, %.4f]\n",
                        param->name, dv, param->lower_bound, param->upper_bound);
                }
            } else {
                fprintf(stderr, "[WARNING] '%s': could not parse 'default='\n", param->name);
            }
        }
    }

    param->usage = USAGE_UNKNOWN;
    param->target_index = -1;

    return 1;
}

static int load_config(const char* filename, Config* config) {
    FILE* file = fopen(filename, "r");
    if(!file) {
        printf("Error: Could not open parameter file: %s\n", filename);
        return 0;
    }

    memset(config, 0, sizeof(Config));

    /* ============================================================
       RUNTIME DEFAULTS — from compile_config.h (Section 2).
       Every field below can still be overridden from
       param_reader.ini if the corresponding key is present.
       ============================================================ */
    config->max_chain_length       = DEF_MAX_CHAIN_LENGTH;
    config->begin_cov_update       = DEF_BEGIN_COV_UPDATE;
    config->update_time            = DEF_UPDATE_TIME;
    config->low_step_bound         = DEF_LOW_STEP_BOUND;
    config->high_step_bound        = DEF_HIGH_STEP_BOUND;
    config->decrease_step          = DEF_DECREASE_STEP;
    config->increase_step          = DEF_INCREASE_STEP;
    config->begin_dragging         = DEF_BEGIN_DRAGGING;
    config->n_drag                 = DEF_N_DRAG;
    config->rmin_points            = DEF_RMIN_POINTS;
    config->rbreak                 = DEF_RBREAK;
    config->min_size_for_freeze_in = DEF_MIN_SIZE_FOR_FREEZE_IN;
    config->max_previous_points    = DEF_MAX_PREVIOUS_POINTS;
    config->initial_drfactor       = DEF_INITIAL_DRFACTOR;

    config->use_dragging           = DEF_USE_DRAGGING;

    config->use_emulator           = DEF_USE_EMULATOR;
    config->emulator_type          = DEF_EMULATOR_TYPE;
    config->poly_degree            = DEF_POLY_DEGREE;
    config->fallback_threshold     = DEF_FALLBACK_THRESHOLD;
    config->train_on_fallback_only = DEF_TRAIN_ON_FALLBACK_ONLY;
    config->emu_bias_min_n         = DEF_EMU_BIAS_MIN_N;
    config->n_recovery             = DEF_N_RECOVERY;
    config->max_emu_rescues        = DEF_MAX_EMU_RESCUES;
    config->emu_bias_max           = DEF_EMU_BIAS_MAX;
    config->free_reject_dchi2_max  = DEF_FREE_REJECT_DCHI2_MAX;
    config->free_reject_ratio      = DEF_FREE_REJECT_RATIO;

    config->verbose                = DEF_VERBOSE;
    config->restart                = DEF_RESTART;
    strncpy(config->outdir, DEF_OUTDIR, sizeof(config->outdir) - 1);
    config->outdir[sizeof(config->outdir) - 1] = '\0';
    strncpy(config->load_cov_path, DEF_LOAD_COV_PATH, sizeof(config->load_cov_path) - 1);
    config->load_cov_path[sizeof(config->load_cov_path) - 1] = '\0';

    for(int i=0; i<MAX_LIKELIHOODS; i++) {
        config->lh_nuisance_exist[i]  = 0;
        config->lh_nuisance_sample[i] = 0;
    }

    config->use_sn          = 0;
    config->sample_nuisance = 1;

    char line[MAX_LINE_LENGTH];

    while(fgets(line, sizeof(line), file)) {
        char* trimmed = trim(line);
        if(strlen(trimmed) == 0 || trimmed[0] == '#' || trimmed[0] == ';') continue;

        if(strstr(trimmed, "CAMB_path") != NULL) continue;
        if(strncmp(trimmed, "cosmological_sample", 19) == 0) {
            fprintf(stderr, "[FATAL ERROR] Obsolete 'cosmological_sample' found. Please update param_reader.ini.\n");
            exit(1);
        }

        /* ============================================================
           GLOBAL FLAGS
           ============================================================ */
        if (strncmp(trimmed, "USE_SN", 6) == 0) {
            char* eq = strchr(trimmed, '=');
            if(eq) sscanf(eq + 1, "%d", &config->use_sn);
            continue;
        }
        if (strncmp(trimmed, "SAMPLE_NUISANCE", 15) == 0) {
            char* eq = strchr(trimmed, '=');
            if(eq) sscanf(eq + 1, "%d", &config->sample_nuisance);
            continue;
        }

        /* ============================================================
           PATHS
           ============================================================ */
        if (strncmp(trimmed, "CAMB_HighL_Template", 19) == 0) {
            char* eq = strchr(trimmed, '=');
            if(eq) strncpy(config->camb_highl_template, trim(eq + 1), 255);
            continue;
        }
        if (strncmp(trimmed, "SN_Data_Path", 12) == 0) {
            char* eq = strchr(trimmed, '=');
            if(eq) strncpy(config->sn_data_path, trim(eq + 1), 255);
            continue;
        }
        if (strncmp(trimmed, "SN_Cov_Path", 11) == 0) {
            char* eq = strchr(trimmed, '=');
            if(eq) strncpy(config->sn_cov_path, trim(eq + 1), 255);
            continue;
        }

        /* ============================================================
           RUNTIME TUNABLES — Sampler
           ============================================================ */
        if (strncmp(trimmed, "MAX_CHAIN_LENGTH",       16) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->max_chain_length);       continue; }
        if (strncmp(trimmed, "BEGIN_COV_UPDATE",       16) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->begin_cov_update);       continue; }
        if (strncmp(trimmed, "UPDATE_TIME",            11) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->update_time);            continue; }
        if (strncmp(trimmed, "LOW_STEP_BOUND",         14) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->low_step_bound);          continue; }
        if (strncmp(trimmed, "HIGH_STEP_BOUND",        15) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->high_step_bound);         continue; }
        if (strncmp(trimmed, "DECREASE_STEP",          13) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%lf",&config->decrease_step);           continue; }
        if (strncmp(trimmed, "INCREASE_STEP",          13) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%lf",&config->increase_step);           continue; }
        if (strncmp(trimmed, "BEGIN_DRAGGING",         14) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->begin_dragging);          continue; }
        if (strncmp(trimmed, "N_DRAG",                  6) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->n_drag);                  continue; }
        if (strncmp(trimmed, "RMIN_POINTS",            11) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->rmin_points);             continue; }
        if (strncmp(trimmed, "RBREAK",                  6) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%lf",&config->rbreak);                  continue; }
        if (strncmp(trimmed, "MIN_SIZE_FOR_FREEZE_IN", 22) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->min_size_for_freeze_in); continue; }
        if (strncmp(trimmed, "MAX_PREVIOUS_POINTS",    19) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->max_previous_points);    continue; }
        if (strncmp(trimmed, "INITIAL_DRFACTOR",       16) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%lf",&config->initial_drfactor);       continue; }

        /* ============================================================
           RUNTIME TUNABLES — Dragging
           ============================================================ */
        if (strncmp(trimmed, "USE_DRAGGING",           12) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->use_dragging);           continue; }

        /* ============================================================
           RUNTIME TUNABLES — Emulator
           ============================================================ */
        if (strncmp(trimmed, "USE_EMULATOR",           12) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->use_emulator);           continue; }
        if (strncmp(trimmed, "EMULATOR_TYPE",          13) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->emulator_type);          continue; }
        if (strncmp(trimmed, "POLY_DEGREE",            11) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->poly_degree);            continue; }
        if (strncmp(trimmed, "FALLBACK_THRESHOLD",     18) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%lf",&config->fallback_threshold);      continue; }
        if (strncmp(trimmed, "TRAIN_ON_FALLBACK_ONLY", 22) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->train_on_fallback_only); continue; }
        if (strncmp(trimmed, "EMU_BIAS_MIN_N",         14) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->emu_bias_min_n);         continue; }
        if (strncmp(trimmed, "N_RECOVERY",             10) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->n_recovery);             continue; }
        if (strncmp(trimmed, "MAX_EMU_RESCUES",        15) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->max_emu_rescues);        continue; }
        if (strncmp(trimmed, "EMU_BIAS_MAX",           12) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%lf",&config->emu_bias_max);            continue; }
        if (strncmp(trimmed, "FREE_REJECT_DCHI2_MAX",  21) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%lf",&config->free_reject_dchi2_max);   continue; }
        if (strncmp(trimmed, "FREE_REJECT_RATIO",      17) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->free_reject_ratio);      continue; }

        /* ============================================================
           RUNTIME TUNABLES — I/O
           ============================================================ */
        if (strncmp(trimmed, "VERBOSE",                 7) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->verbose);               continue; }
        if (strncmp(trimmed, "RESTART",                 7) == 0) { char* eq=strchr(trimmed,'='); if(eq) sscanf(eq+1,"%d", &config->restart);               continue; }
        if (strncmp(trimmed, "OUTDIR",                  6) == 0) { char* eq=strchr(trimmed,'='); if(eq) strncpy(config->outdir, trim(eq+1), 511);         continue; }
        if (strncmp(trimmed, "LOAD_COV_PATH",          13) == 0) { char* eq=strchr(trimmed,'='); if(eq) strncpy(config->load_cov_path, trim(eq+1), 511);   continue; }

        /* ============================================================
           LIKELIHOOD PATHS
           ============================================================ */
        int lh_idx = 0;
        if (sscanf(trimmed, "Likelihood_Path_%d", &lh_idx) == 1) {
            char* eq = strchr(trimmed, '=');
            if(eq && lh_idx > 0 && lh_idx <= MAX_LIKELIHOODS) {
                strncpy(config->likelihood_paths[lh_idx - 1], trim(eq + 1), 255);
                if (lh_idx > config->likelihood_count) config->likelihood_count = lh_idx;
            }
            continue;
        }

        /* LH_N_nuisance_{exist,sample}: capture subtype to avoid false match */
        {
            int      nuis_idx         = 0;
            char     nuis_subtype[32] = "";
            if (sscanf(trimmed, "LH_%d_nuisance_%31s", &nuis_idx, nuis_subtype) == 2 &&
                nuis_idx > 0 && nuis_idx <= MAX_LIKELIHOODS) {
                char* eq = strchr(trimmed, '=');
                if (eq) {
                    int val = 0;
                    if (sscanf(eq + 1, "%d", &val) != 1) val = 0;

                    if (strcmp(nuis_subtype, "exist") == 0) {
                        config->lh_nuisance_exist[nuis_idx - 1] = val;
                        printf("[CONFIG] LH_%d_nuisance_exist  = %d\n", nuis_idx, val);
                    } else if (strcmp(nuis_subtype, "sample") == 0) {
                        config->lh_nuisance_sample[nuis_idx - 1] = val;
                        printf("[CONFIG] LH_%d_nuisance_sample = %d\n", nuis_idx, val);
                    } else {
                        fprintf(stderr, "[WARNING] Unknown key LH_%d_nuisance_%s; ignored\n",
                                nuis_idx, nuis_subtype);
                    }
                }
                continue;
            }
        }

        if(strstr(trimmed, "=") != NULL && config->param_count < MAX_PARAMS) {
            if(parse_parameter_line(trimmed, &config->params[config->param_count])) {
                config->param_count++;
            }
        }
    }
    fclose(file);

    /* SAMPLE_NUISANCE=0: nuisances → fixed (uses default= or midpoint) */
    if (config->sample_nuisance == 0) {
        for (int i = 0; i < config->param_count; i++) {
            ParameterConfig *p = &config->params[i];
            if (p->class_type == PARAM_CLASS_NUISANCE && p->state == PARAM_SAMPLED) {
                p->state = PARAM_FIXED;
                p->is_estimated = 0;

                if (p->has_default) {
                    p->fixed_value = p->default_value;
                    printf("[INFO] SAMPLE_NUISANCE=0: '%s' fixed at %.4f (default=)\n",
                           p->name, p->fixed_value);
                } else {
                    p->fixed_value = 0.5 * (p->lower_bound + p->upper_bound);
                    printf("[INFO] SAMPLE_NUISANCE=0: '%s' fixed at %.4f (midpoint)\n",
                           p->name, p->fixed_value);
                }
            }
        }
    }

    /* USE_SN=0: M_B → fixed (uses default= or -19.3) */
    if (config->use_sn == 0) {
        for (int i = 0; i < config->param_count; i++) {
            ParameterConfig *p = &config->params[i];
            if (strcmp(p->name, "M_B") == 0 && p->state == PARAM_SAMPLED) {
                p->state = PARAM_FIXED;
                p->is_estimated = 0;

                if (p->has_default) {
                    p->fixed_value = p->default_value;
                    printf("[INFO] USE_SN=0: '%s' fixed at %.4f (default=)\n",
                           p->name, p->fixed_value);
                } else {
                    p->fixed_value = -19.3;
                    printf("[INFO] USE_SN=0: '%s' fixed at %.4f (hardcoded fallback)\n",
                           p->name, p->fixed_value);
                }
            }
        }
    }

    for (int i=0; i<config->param_count; i++) {
        ParameterConfig* p = &config->params[i];
        if (p->state == PARAM_SAMPLED) {
            config->num_sampled_total++;
            if (p->class_type == PARAM_CLASS_COSMO) config->num_cosmo_sampled++;
            else config->num_nuis_sampled++;

            if (p->speed == PARAM_SPEED_SLOW) config->num_slow_sampled++;
            else config->num_fast_sampled++;
        } else {
            if (p->class_type == PARAM_CLASS_COSMO) config->num_cosmo_fixed++;
            else config->num_nuis_fixed++;
        }
    }

    printf("\n========================================================================\n");
    printf("                  MCMC CONFIGURATION SUMMARY\n");
    printf("========================================================================\n");
    printf("Total parameters defined       : %d\n", config->param_count);
    printf("Cosmological parameters defined: %d\n", config->num_cosmo_sampled + config->num_cosmo_fixed);
    printf("Nuisance parameters defined    : %d\n", config->num_nuis_sampled + config->num_nuis_fixed);
    printf("\n");
    printf("Sampled cosmological parameters: %d\n", config->num_cosmo_sampled);
    printf("Sampled nuisance parameters    : %d\n", config->num_nuis_sampled);
    printf("\n");
    printf("Fixed cosmological parameters  : %d\n", config->num_cosmo_fixed);
    printf("Fixed nuisance parameters      : %d\n", config->num_nuis_fixed);
    printf("\n");
    printf("Sampled slow parameters        : %d\n", config->num_slow_sampled);
    printf("Sampled fast parameters        : %d\n", config->num_fast_sampled);
    printf("\n");
    printf("Total MCMC dimensions          : %d\n", config->num_sampled_total);
    printf("\n--- Runtime tunables ---\n");
    printf("USE_EMULATOR           : %d\n", config->use_emulator);
    printf("USE_DRAGGING           : %d\n", config->use_dragging);
    printf("VERBOSE                : %d\n", config->verbose);
    printf("MAX_CHAIN_LENGTH       : %d\n", config->max_chain_length);
    printf("N_DRAG                 : %d\n", config->n_drag);
    printf("INITIAL_DRFACTOR       : %.4f\n", config->initial_drfactor);
    printf("OUTDIR                 : %s\n", config->outdir);
    if (config->load_cov_path[0])
        printf("LOAD_COV_PATH          : %s\n", config->load_cov_path);
    printf("========================================================================\n\n");

    return 1;
}

static double compute_prior_penalty(double* theta) {
    double penalty = 0.0;
    int param_idx = 0;
    for (int i = 0; i < global_config.param_count; i++) {
        if (global_config.params[i].state == PARAM_SAMPLED) {
            if (global_config.params[i].prior_type == PRIOR_GAUSSIAN) {
                double val = theta[param_idx];
                double mean = global_config.params[i].gaussian_mean;
                double sig = global_config.params[i].gaussian_sigma;
                double diff = (val - mean) / sig;
                penalty += diff * diff;
            }
            param_idx++;
        }
    }
    return penalty;
}
#endif
