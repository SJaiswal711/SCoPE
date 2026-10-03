#ifndef _PARAM_CONFIG_H_
#define _PARAM_CONFIG_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "clik.h"

#define MAX_PARAMS 200
#define MAX_LINE_LENGTH 512
#define MAX_PARAM_NAME_LENGTH 100
#define MAX_LIKELIHOODS 10

// --- ENUMS & STRUCTS ---

typedef enum {
    PARAM_CLASS_COSMO,
    PARAM_CLASS_NUISANCE
} ParameterClass;

typedef enum {
    PARAM_SPEED_SLOW,
    PARAM_SPEED_FAST
} ParameterSpeed;

typedef enum {
    PARAM_FIXED,
    PARAM_SAMPLED
} ParameterState;

typedef enum {
    USAGE_UNKNOWN = 0,
    USAGE_CAMB,      
    USAGE_NUISANCE, 
    USAGE_DERIVED  
} ParamUsage;

typedef enum {
    PRIOR_FLAT,
    PRIOR_GAUSSIAN
} PriorType;

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

    int is_estimated; // Derived: state == PARAM_SAMPLED

    ParamUsage usage; 
    int target_index; 
} ParameterConfig;

typedef struct {
    char likelihood_paths[MAX_LIKELIHOODS][256];
    int lh_nuisance_exist[MAX_LIKELIHOODS];
    int lh_nuisance_sample[MAX_LIKELIHOODS];
    int likelihood_count;

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

// --- HELPER FUNCTIONS ---

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
    for(int i=0; i<MAX_LIKELIHOODS; i++) {
        config->lh_nuisance_exist[i] = 0;
        config->lh_nuisance_sample[i] = 0;
    }

    char line[MAX_LINE_LENGTH];

    while(fgets(line, sizeof(line), file)) {
        char* trimmed = trim(line);
        if(strlen(trimmed) == 0 || trimmed[0] == '#' || trimmed[0] == ';') continue;

        if(strstr(trimmed, "CAMB_path") != NULL) continue;
        if(strncmp(trimmed, "cosmological_sample", 19) == 0) {
            fprintf(stderr, "[FATAL ERROR] Obsolete 'cosmological_sample' found. Please update param_reader.ini to the new explicit formatting structure.\n");
            exit(1);
        }

        int lh_idx = 0;
        if (sscanf(trimmed, "Likelihood_Path_%d", &lh_idx) == 1) {
            char* eq = strchr(trimmed, '=');
            if(eq && lh_idx > 0 && lh_idx <= MAX_LIKELIHOODS) {
                strncpy(config->likelihood_paths[lh_idx - 1], trim(eq + 1), 255);
                if (lh_idx > config->likelihood_count) config->likelihood_count = lh_idx;
            }
            continue;
        }
        if (sscanf(trimmed, "LH_%d_nuisance_exist", &lh_idx) == 1) {
            char* eq = strchr(trimmed, '=');
            if(eq && lh_idx > 0 && lh_idx <= MAX_LIKELIHOODS) sscanf(eq + 1, "%d", &config->lh_nuisance_exist[lh_idx - 1]);
            continue;
        }
        if (sscanf(trimmed, "LH_%d_nuisance_sample", &lh_idx) == 1) {
            char* eq = strchr(trimmed, '=');
            if(eq && lh_idx > 0 && lh_idx <= MAX_LIKELIHOODS) sscanf(eq + 1, "%d", &config->lh_nuisance_sample[lh_idx - 1]);
            continue;
        }

        if(strstr(trimmed, "=") != NULL && config->param_count < MAX_PARAMS) {
            if(parse_parameter_line(trimmed, &config->params[config->param_count])) {
                config->param_count++;
            }
        }
    }
    fclose(file);

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
                penalty += diff * diff; // Converts Gaussian prior to a Delta Chi-square penalty
            }
            param_idx++;
        }
    }
    return penalty;
}
#endif
