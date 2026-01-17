# IMPORTING THE REQUIRES LIBRARIES
import pandas as pd
import pymc as pm
from scipy import stats as sts
import numpy as np
import pytensor.tensor as pt
import pytensor
pytensor.config.cxx = "/usr/bin/clang++"

# IMPORTING THE DATA

data_path = "/Users/fabian/Documents/epidemiology_downloads/downloads_set_1/"
df_poblacion = pd.read_csv(data_path + "DatosPoblacionDpto.csv")
df_reportados = pd.read_csv(data_path + "confirmado_diarios_revisado.csv")
df_registro_diario = pd.read_csv(data_path + "REGISTRO DIARIO_Datos completos_data.csv")
df_fallecidos = pd.read_csv(data_path + "Fallecidos_diarios_revisado.csv")
df_inmunizados = pd.read_csv(data_path + "Inmunizado_diarios.csv")


# total population
N_total_population = df_poblacion["2020"].sum()

# daily reported cases
reportados_daily = df_reportados["Confirmado_diario"].values

df_registro_diario = df_registro_diario.fillna(0) # Replicate NA=0 from R scripts

# imported covid cases
importados_daily = df_registro_diario["Confirmados en albergues"].values

# hospitalized (H)
hospitalized_current = df_registro_diario["Internados Generales"].values

# in ICU (U)
icu_current = df_registro_diario["Internados UTI"].values

# deaths (F)
fallecidos_daily = df_fallecidos["Fallecido_diario"].values

# immunization
inmunizados_daily = df_inmunizados["Inmunizado_diario"].fillna(0).values

# initial parameter estimation
w = 14
data_imported = importados_daily[0:w]
data_vaccinated = inmunizados_daily[0:w]
observed_daily_cases = reportados_daily[0:w]
t_points = np.arange(1, w + 1) # days


# previously known values 
alpha     = 1.0 / 3.0 # reciprocal of average latent period
gamma     = 1.0 / 7.0 # reciprocal of average infectious period
delta_hu  = 1.0 / 7.0 # average stay in hospital period
delta_hf  = 1.0 / 9.0 # average stay in hospital bed
delta_ho  = 1.0 / 11.0 # average recovery period in hospital bed
phi_uf    = 1.0 / 11.0 # average stay in ICU before death
phi_uo    = 1.0 / 12.0 # average recovery period in ICU
psi       = 1.0 / 360.0
eta       = 0.9

def create_ode_func(imported, vaccinated, N, alpha, gamma, psi, eta, w):
    """
    This factory takes our static data and returns the actual
    ODE function that the solver will use.
    """
    # converting into pytensor variable
    imported_fixed = pt.as_tensor_variable(imported)
    vaccinated_fixed = pt.as_tensor_variable(vaccinated)

    def ode_func_with_data(y, t, p):
        """
        This is the actual function to be solved.
        'p' will *only* contain the one parameter we estimate: beta.
        """
        
        # unpack the array
        S, E, I, R, O = y[0], y[1], y[2], y[3], y[4]
        beta = p[0]
        
        # making sure we only have integers as indices 
        index_raw = pt.cast(pt.ceil(t), 'int32') - 1
        index = pt.clip(index_raw, 0, w - 1)
        imp = imported_fixed[index]
        vac = vaccinated_fixed[index]

        # ODEs for the disease dynamics
        dS_dt = -beta * I * S / N + psi * O - eta * vac - imp
        dE_dt =  beta * I * S / N - alpha * E
        dI_dt =  alpha * E - gamma * I
        dR_dt =  gamma * I + imp
        dO_dt =  gamma * I + imp - psi * O + eta * vac
        
        return [dS_dt, dE_dt, dI_dt, dR_dt, dO_dt]
    
    # Return the new function
    return ode_func_with_data


import arviz as az
az.plot_trace(idata)

az.plot_rank(idata)

# 1. Get the log-posterior values from the inference data
log_posterior = idata.sample_stats.lp

# 2. Find the exact location (chain and draw) of the highest value
# We use unravel_index to convert the "flat" index from argmax
# into (chain, draw) coordinates.
max_lp_coords = np.unravel_index(
    log_posterior.values.argmax(), 
    log_posterior.shape
)

# 3. Extract the "best" parameters from that single sample
# We use the coordinates to pull the values from the posterior
best_params = idata.posterior.isel(
    chain=max_lp_coords[0], 
    draw=max_lp_coords[1]
)

# 4. Get the final values
best_beta = best_params["beta"].item()
best_e0 = best_params["e0"].item()
best_i0 = best_params["i0"].item()

print(f"Best Beta: {best_beta}")
print(f"Best e0:   {best_e0}")
print(f"Best i0:   {best_i0}")
