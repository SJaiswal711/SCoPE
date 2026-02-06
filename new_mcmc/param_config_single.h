#ifndef _PARAM_CONFIG_SINGLE_H_
#define _PARAM_CONFIG_SINGLE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "clik.h" 

#define MAX_PARAMS 100
#define MAX_LINE_LENGTH 256
#define MAX_PARAM_NAME_LENGTH 100

// --- ENUMS & STRUCTS ---

typedef enum {
    USAGE_UNKNOWN = 0,
    USAGE_CAMB,      
    USAGE_NUISANCE, 
    USAGE_DERIVED  
} ParamUsage;

typedef struct {
    char name[MAX_PARAM_NAME_LENGTH];
    double lower_bound;
    double upper_bound;
    double sigma;
    int is_estimated;
    
    ParamUsage usage; 
    int target_index; 
} ParameterConfig;

typedef struct {
    char camb_path[256];
    char likelihood_paths[10][256];
    int likelihood_count;
    ParameterConfig params[MAX_PARAMS];
    int param_count;
} Config;

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
           name[i] != '^' && name[i] != '\\') {
            clean_name[j++] = name[i];
        }
    }
    clean_name[j] = '\0';
    
    strncpy(param->name, clean_name, MAX_PARAM_NAME_LENGTH - 1);
    param->name[MAX_PARAM_NAME_LENGTH - 1] = '\0';
    
    int parsed = sscanf(value_str, "%lf %lf %lf", 
                       &param->lower_bound, &param->upper_bound, &param->sigma);
    
    param->is_estimated = (parsed == 3);
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
    config->param_count = 0;
    config->likelihood_count = 0;
    
    char line[MAX_LINE_LENGTH];
    
    while(fgets(line, sizeof(line), file)) {
        char* trimmed = trim(line);
        if(strlen(trimmed) == 0 || trimmed[0] == '#' || trimmed[0] == ';') continue;
        
        if(strstr(trimmed, "CAMB_path") != NULL) continue;
        if(strstr(trimmed, "Likelihood_Path_") != NULL) {
            char* equals_pos = strchr(trimmed, '=');
            if(equals_pos && config->likelihood_count < 10) {
                strncpy(config->likelihood_paths[config->likelihood_count], 
                       trim(equals_pos + 1), sizeof(config->likelihood_paths[0]) - 1);
                config->likelihood_count++;
            }
            continue;
        }
        
        if(strchr(trimmed, '$') != NULL || (strstr(trimmed, "=") != NULL && config->param_count < MAX_PARAMS)) {
            if(parse_parameter_line(trimmed, &config->params[config->param_count])) {
                config->param_count++;
            }
        }
    }
    fclose(file);
    return 1;
}

// --- AUTO-MAPPING FUNCTION ---
static void map_parameters(Config* config, clik_object* clikid) {
    
    parname *nuis_names;
    error *err = initError();
    int n_nuis = clik_get_extra_parameter_names(clikid, &nuis_names, &err);
    if (isError(err)) {
        fprintf(stderr, "Error getting clik names.\n");
        return;
    }

    const char* camb_names[] = {
        "Omega_m_h2", "Omega_b_h2", "h", "tau", "n_s", "A_s", 
        "w", "wa", "mnu", "nrun", "Neff", "omk"
    };
    int n_camb_names = 12;

    printf("\n--- Performing Parameter Mapping ---\n");
    
    // DEBUG: Print what Clik wants
    printf("DEBUG: The Likelihood (clik) requested the following %d parameters:\n", n_nuis);
    for(int k=0; k<n_nuis; k++) {
        printf("   - [%d] '%s'\n", k, nuis_names[k]);
    }
    printf("------------------------------------\n");

    for(int i=0; i < config->param_count; i++) {
        ParameterConfig* p = &config->params[i];
        p->usage = USAGE_UNKNOWN;

        for(int j=0; j<n_nuis; j++) {
            // Trim comparison just in case
            if(strcmp(p->name, nuis_names[j]) == 0) {
                p->usage = USAGE_NUISANCE;
                p->target_index = j;
                printf("  [MAPPED] %s -> Nuisance Vector Index %d\n", p->name, j);
                break;
            }
        }
        if(p->usage != USAGE_UNKNOWN) continue;

        for(int j=0; j<n_camb_names; j++) {
            if(strcmp(p->name, camb_names[j]) == 0) {
                p->usage = USAGE_CAMB;
                p->target_index = j;
                printf("  [MAPPED] %s -> CAMB Vector Index %d\n", p->name, j);
                break;
            }
        }
        
        if(p->usage == USAGE_UNKNOWN && p->is_estimated) {
            printf("  [WARNING] Parameter '%s' is estimated but NOT mapped to CAMB or Likelihood!\n", p->name);
        }
    }
    free(nuis_names);
    printf("------------------------------------\n\n");
}

#endif