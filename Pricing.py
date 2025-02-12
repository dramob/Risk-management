import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm

# Configuration de la page
st.set_page_config(page_title="Pricing d'Options - Projet 3", layout="wide")
st.title("Projet 3 : Pricing d'Options Monte Carlo")
st.markdown("""
Cette application simule le pricing d’options sur actions via une méthode Monte Carlo.
Vous pouvez choisir parmi différents types d’options :
- **Vanilla Call**
- **Vanilla Put**
- **Tunnel Option** (le payoff est versé uniquement si la trajectoire reste entre des bornes définies)
- **Himalaya Option** (basée sur le maximum atteint par le sous-jacent)
- **Napoléon Option** (basée sur le minimum atteint par le sous-jacent)
""")

# --------------------------
# Formulaire de paramètres de l'option et de la simulation
# --------------------------
with st.form("option_params"):
    col1, col2 = st.columns(2)
    with col1:
        S0 = st.number_input("Prix initial de l'action (S0)", value=100.0, step=1.0)
        r = st.number_input("Taux sans risque (r)", value=0.05, step=0.001, format="%.3f")
        sigma = st.number_input("Volatilité (σ)", value=0.2, step=0.01, format="%.2f")
    with col2:
        T = st.number_input("Maturité (T en années)", value=1.0, step=0.1, format="%.1f")
        steps = st.number_input("Nombre de pas de temps", value=252, step=1)
        N_sim = st.number_input("Nombre de simulations", value=10000, step=1000)
    
    option_type = st.selectbox("Type d'option", 
                                ("Vanilla Call", "Vanilla Put", "Tunnel Option", "Himalaya Option", "Napoléon Option"))
    K = st.number_input("Prix d'exercice (Strike, K)", value=100.0, step=1.0)
    
    # Paramètres spécifiques pour Tunnel Option
    if option_type == "Tunnel Option":
        lower_barrier = st.number_input("Borne inférieure du tunnel", value=80.0, step=1.0)
        upper_barrier = st.number_input("Borne supérieure du tunnel", value=120.0, step=1.0)
    else:
        lower_barrier, upper_barrier = None, None
    
    submit = st.form_submit_button("Lancer la simulation")

# --------------------------
# Simulation des trajectoires et calcul des payoffs
# --------------------------
if submit:
    dt = T / steps  # pas de temps
    # Pré-allocation d'un tableau pour stocker les trajectoires
    paths = np.zeros((int(N_sim), int(steps) + 1))
    paths[:, 0] = S0
    
    # Simulation vectorisée des trajectoires
    for t in range(1, int(steps)+1):
        Z = np.random.normal(0, 1, int(N_sim))
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Calcul du payoff en fonction du type d'option
    if option_type == "Vanilla Call":
        payoffs = np.maximum(paths[:, -1] - K, 0)
    elif option_type == "Vanilla Put":
        payoffs = np.maximum(K - paths[:, -1], 0)
    elif option_type == "Tunnel Option":
        # Pour la Tunnel Option, le payoff (similaire à un Call) est versé si toute la trajectoire reste dans [lower_barrier, upper_barrier]
        within_tunnel = np.all((paths >= lower_barrier) & (paths <= upper_barrier), axis=1)
        payoffs = np.where(within_tunnel, np.maximum(paths[:, -1] - K, 0), 0)
    elif option_type == "Himalaya Option":
        # Option basée sur le maximum atteint par le sous-jacent
        max_prices = np.max(paths, axis=1)
        payoffs = np.maximum(max_prices - K, 0)
    elif option_type == "Napoléon Option":
        # Option basée sur le minimum atteint par le sous-jacent
        min_prices = np.min(paths, axis=1)
        payoffs = np.maximum(K - min_prices, 0)
    else:
        payoffs = np.zeros(int(N_sim))
    
    # Actualisation des payoffs
    discounted_payoffs = np.exp(-r * T) * payoffs
    option_price = np.mean(discounted_payoffs)
    
    # Calcul de l'erreur à 99% (utilisation de la valeur z pour 99% de confiance)
    z_99 = norm.ppf(0.995)  # Pour un intervalle à 99%
    std_error = np.std(discounted_payoffs) / np.sqrt(N_sim)
    error_99 = z_99 * std_error
    
    # Convergence : moyenne cumulée des payoffs actualisés
    cum_avg = np.cumsum(discounted_payoffs) / np.arange(1, int(N_sim)+1)
    convergence_df = pd.DataFrame({
        "Simulation": np.arange(1, int(N_sim)+1),
        "Prix moyen cumulée": cum_avg
    })
    
    # Affichage des résultats
    st.subheader("Résultats de la Simulation")
    st.write(f"**Prix de l'option ({option_type})** : {option_price:,.2f} €")
    st.write(f"**Erreur à 99%** : ± {error_99:,.2f} €")
    
    # Graphique de convergence du prix moyen
    conv_chart = alt.Chart(convergence_df).mark_line().encode(
        x=alt.X("Simulation:Q", title="Nombre de simulations"),
        y=alt.Y("Prix moyen cumulée:Q", title="Prix moyen (€)")
    ).properties(title="Convergence du prix moyen de l'option")
    st.altair_chart(conv_chart, use_container_width=True)
    
    # Histogramme de la distribution des payoffs actualisés
    payoff_df = pd.DataFrame({"Payoff actualisé": discounted_payoffs})
    hist_chart = alt.Chart(payoff_df).mark_bar().encode(
        alt.X("Payoff actualisé:Q", bin=alt.Bin(maxbins=30), title="Payoff actualisé (€)"),
        alt.Y("count()", title="Fréquence")
    ).properties(title="Distribution des payoffs actualisés")
    st.altair_chart(hist_chart, use_container_width=True)