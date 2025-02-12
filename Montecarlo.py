import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import norm

# --------------------------
# Chargement et mise en cache du fichier Excel
# --------------------------
@st.cache_data
def load_excel(file):
    try:
        data = pd.read_excel(file, sheet_name=None, engine="openpyxl")
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier Excel : {e}")
        return None

# --------------------------
# Extraction du mapping PD depuis la feuille "Params"
# --------------------------
def extract_pd_mapping(data):
    """
    Extrait la table des PD depuis la feuille "Params".
    On suppose que les lignes 2 à 20 (indices 1 à 19) et les colonnes E à H (indices 4 à 7) contiennent :
      [Notation, PD_1Y, PD_3Y, PD_5Y]
    Retourne un dictionnaire de la forme :
      { "Aaa": {"1Y": valeur, "3Y": valeur, "5Y": valeur}, ... }
    """
    if "Params" not in data:
        st.error("La feuille 'Params' n'a pas été trouvée dans le fichier Excel.")
        return {}
    params_df = data["Params"]
    params_df.columns = params_df.columns.astype(str).str.strip()
    pd_table = params_df.iloc[1:20, 4:8].copy()
    pd_table.columns = ["Notation", "1Y", "3Y", "5Y"]
    
    def clean_percentage(val):
        if isinstance(val, str):
            val = val.replace(",", ".").replace("%", "").strip()
        try:
            return float(val) / 100.0
        except Exception:
            return None
    
    for col in ["1Y", "3Y", "5Y"]:
        pd_table[col] = pd_table[col].apply(clean_percentage)
    
    pd_mapping = pd_table.set_index("Notation").to_dict("index")
    return pd_mapping

# --------------------------
# Simulation Monte Carlo du portefeuille
# --------------------------
def simulate_portfolio_loss(portfolio_df, N, rho_global, rho_sector, pd_mapping, horizon):
    """
    Pour chaque crédit du portefeuille, on assigne une PD à partir du mapping selon sa Rating.
    Pour un horizon donné (idéalement 1, 3 ou 5 ans), on sélectionne la PD correspondante. Si la notation n'est pas trouvée, on utilise une valeur par défaut.
    
    On simule ensuite N scénarios à l'aide d'un modèle gaussien à deux facteurs :
      Z = sqrt(rho_global)*X + sqrt(rho_sector)*Y + sqrt(1 - rho_global - rho_sector)*eps
    Un crédit est en défaut si Z < B, où B = norm.ppf(PD).
    
    La perte pour un crédit en défaut est Exposure * LGD.
    La perte totale du portefeuille est la somme de ces pertes pour chaque simulation.
    """
    n_credits = portfolio_df.shape[0]
    
    # Pour chaque crédit, obtenir la PD correspondant au horizon choisi
    PDs = []
    for idx, row in portfolio_df.iterrows():
        rating = str(row["Rating"]).strip()
        # Choix de l'horizon : 1Y, 3Y ou 5Y (si horizon non conforme, on prend 3Y par défaut)
        if abs(horizon - 1) < 0.1:
            key = "1Y"
        elif abs(horizon - 3) < 0.1:
            key = "3Y"
        elif abs(horizon - 5) < 0.1:
            key = "5Y"
        else:
            key = "3Y"
        pd_val = pd_mapping.get(rating, {}).get(key, 0.0001)
        PDs.append(pd_val)
    PDs = np.array(PDs)  # vecteur de taille n_credits

    # Extraire LGD et Exposure
    LGDs = []
    Exposures = []
    for idx, row in portfolio_df.iterrows():
        # Conversion de LGD (si nécessaire)
        lgd_val = row["LGD"]
        if isinstance(lgd_val, str):
            lgd_val = float(lgd_val.replace("%", "").strip()) / 100.0
        else:
            lgd_val = float(lgd_val)
        LGDs.append(lgd_val)
        Exposures.append(float(row["Exposure"]))
    LGDs = np.array(LGDs)
    Exposures = np.array(Exposures)
    
    # Total notional du portefeuille
    total_notional = np.sum(Exposures)
    
    # Simulation des facteurs communs
    X = np.random.normal(0, 1, size=(N, 1))
    Y = np.random.normal(0, 1, size=(N, 1))
    eps = np.random.normal(0, 1, size=(N, n_credits))
    loading_idio = np.sqrt(max(0, 1 - rho_global - rho_sector))
    
    # Calcul de la variable latente pour chaque crédit et simulation
    Z = np.sqrt(rho_global) * X + np.sqrt(rho_sector) * Y + loading_idio * eps  # dimension (N, n_credits)
    
    # Calcul des barrières de défaut pour chaque crédit
    # Pour chaque crédit, B = norm.ppf(PD)
    B = norm.ppf(PDs)  # vecteur de taille n_credits
    
    # Comparaison : chaque simulation, pour chaque crédit, défaut si Z < B
    default_indicator = (Z < B)  # matrice booléenne de taille (N, n_credits)
    
    # Calcul de la perte pour chaque crédit en cas de défaut
    credit_loss = np.outer(np.ones(N), Exposures * LGDs)  # dimension (N, n_credits)
    
    # Matrice des pertes effectives
    loss_matrix = default_indicator.astype(float) * credit_loss
    
    # Pertes totales du portefeuille pour chaque simulation
    portfolio_losses = loss_matrix.sum(axis=1)
    
    return portfolio_losses, total_notional

# --------------------------
# Calcul des indicateurs de risque
# --------------------------
def compute_risk_measures(losses, confidence):
    expected_loss = np.mean(losses)
    VaR = np.quantile(losses, confidence)
    ES = np.mean(losses[losses >= VaR])
    return expected_loss, VaR, ES

# --------------------------
# Configuration de la page Streamlit
# --------------------------
st.set_page_config(page_title="Gestion de Portefeuilles & Dérivés de Crédit", layout="wide")
st.title("Projet 2 : Gestion de Portefeuilles et Simulation Monte Carlo")
st.markdown("""
Cette application simule les pertes d’un portefeuille de crédits à l’aide d’un modèle gaussien à deux facteurs et calcule :
- **Expected Loss (EL)**
- **Value at Risk (VaR)**
- **Expected Shortfall (ES)**
  
De plus, elle estime la perte sur une tranche de CDO en fonction des seuils d’attachement et de détarachement.
""")

# --------------------------
# Chargement du fichier Excel dans la barre latérale
# --------------------------
st.sidebar.header("Chargement du fichier Excel")
uploaded_file = st.sidebar.file_uploader("Charger le fichier 'credit.xlsx'", type=["xlsx"])

# --------------------------
# Formulaire des paramètres de simulation
# --------------------------
st.markdown("## Paramètres de Simulation")
with st.form("sim_params"):
    N = st.number_input("Nombre de simulations", value=10000, step=1000)
    confidence = st.slider("Niveau de confiance pour VaR/ES", min_value=0.90, max_value=0.99, value=0.99, step=0.01)
    rho_global = st.number_input("Coefficient de corrélation globale (rho_global)", value=0.2, step=0.01, format="%.2f")
    rho_sector = st.number_input("Coefficient de corrélation sectorielle (rho_sector)", value=0.1, step=0.01, format="%.2f")
    horizon = st.number_input("Horizon (années) pour PD", value=3.0, step=0.5)
    
    st.markdown("### Paramètres de la tranche de CDO")
    attachment = st.number_input("Seuil d'attachement (en % du total, exprimé en décimal)", value=0.03, step=0.01, format="%.2f")
    detachment = st.number_input("Seuil de détarachement (en % du total, exprimé en décimal)", value=0.07, step=0.01, format="%.2f")
    
    submit_sim = st.form_submit_button("Lancer la simulation")

# --------------------------
# Exécution de la simulation si le fichier est chargé et le formulaire soumis
# --------------------------
if submit_sim:
    if uploaded_file is None:
        st.error("Veuillez charger le fichier Excel 'credit.xlsx' via la barre latérale.")
    else:
        data = load_excel(uploaded_file)
        if data is None or "Portfolio" not in data:
            st.error("La feuille 'Portfolio' n'a pas été trouvée dans le fichier Excel.")
        else:
            # Extraction du mapping PD depuis la feuille "Params"
            pd_mapping = extract_pd_mapping(data)
            
            # Récupération de la feuille Portfolio
            portfolio_df = data["Portfolio"]
            portfolio_df.columns = portfolio_df.columns.str.strip()
            # On suppose que la feuille Portfolio contient : Id, Exposure, Rating, LGD, Sector
            
            # Simulation des pertes du portefeuille
            losses, total_notional = simulate_portfolio_loss(portfolio_df, int(N), rho_global, rho_sector, pd_mapping, horizon)
            
            # Calcul des indicateurs de risque
            expected_loss, VaR, ES = compute_risk_measures(losses, confidence)
            
            # Calcul de la convergence de la perte moyenne cumulative
            cum_avg = np.cumsum(losses) / np.arange(1, len(losses)+1)
            cum_avg_df = pd.DataFrame({
                "Simulation": np.arange(1, len(losses)+1),
                "Perte moyenne cumulative": cum_avg
            })
            
            # Calcul de la perte sur la tranche de CDO
            loss_pct = losses / total_notional  # perte en pourcentage
            tranche_losses = np.minimum(np.maximum(loss_pct - attachment, 0), detachment - attachment)
            expected_tranche_loss = np.mean(tranche_losses)
            
            # Affichage des résultats
            st.subheader("Résultats de la Simulation")
            st.write(f"**Expected Loss (portefeuille) :** {expected_loss:,.2f} €")
            st.write(f"**VaR (à {confidence*100:.0f}%) :** {VaR:,.2f} €")
            st.write(f"**Expected Shortfall (ES) :** {ES:,.2f} €")
            st.write(f"**Total Notional du portefeuille :** {total_notional:,.2f} €")
            st.write(f"**Expected Loss de la tranche CDO (attachement = {attachment*100:.0f}%, détarachement = {detachment*100:.0f}%) :** {expected_tranche_loss*100:.2f}% du portefeuille")
            
            # Graphique de convergence de la perte moyenne cumulative
            st.subheader("Convergence de la perte moyenne cumulative")
            conv_chart = alt.Chart(cum_avg_df).mark_line().encode(
                x=alt.X("Simulation:Q", title="Nombre de simulations"),
                y=alt.Y("Perte moyenne cumulative:Q", title="Perte moyenne (€)")
            ).properties(title="Convergence de la perte moyenne cumulative")
            st.altair_chart(conv_chart, use_container_width=True)
            
            # Histogramme de la distribution des pertes du portefeuille
            st.subheader("Distribution des pertes du portefeuille")
            hist_data = pd.DataFrame({"Pertes": losses})
            hist_chart = alt.Chart(hist_data).mark_bar().encode(
                alt.X("Pertes:Q", bin=alt.Bin(maxbins=30), title="Pertes (€)"),
                alt.Y("count()", title="Fréquence")
            ).properties(title="Histogramme des pertes du portefeuille")
            st.altair_chart(hist_chart, use_container_width=True)