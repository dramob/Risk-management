import streamlit as st
import pandas as pd
import altair as alt

# --------------------------
# Chargement et mise en cache du fichier Excel
# --------------------------
@st.cache_data
def load_excel(file):
    """
    Charge le fichier Excel et retourne un dictionnaire contenant toutes les feuilles.
    """
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
    Extrait la table des PD depuis la feuille "Params". On suppose que :
      - Les lignes 2 à 20 (index 1 à 19) contiennent la table.
      - Les colonnes E à H (index 4 à 7) contiennent : [Notation, PD_1Y, PD_3Y, PD_5Y]
    Retourne un dictionnaire du type :
      { "Aaa": {"1Y": value, "3Y": value, "5Y": value}, ... }
    """
    if "Params" not in data:
        st.error("La feuille 'Params' n'a pas été trouvée dans le fichier Excel.")
        return {}
    params_df = data["Params"]
    # Normaliser les noms de colonnes (même si on n'en a pas besoin ici)
    params_df.columns = params_df.columns.astype(str).str.strip()
    # Extraction des lignes 2 à 20 et colonnes E à H
    pd_table = params_df.iloc[1:20, 4:8].copy()
    pd_table.columns = ["Notation", "1Y", "3Y", "5Y"]
    
    # Fonction de nettoyage : convertir des pourcentages (ex. "0.003%" ou "0,003%") en décimal
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
# Fonction de calcul pour une opération de crédit
# --------------------------
def calculate_credit(credit, interest_rate, maturity, pd_mapping):
    """
    Calcule et retourne les indicateurs pour une opération de crédit.
    Utilise :
      - Exposure (montant exposé)
      - Taux d'intérêt et échéance fournis par l'utilisateur
      - Le mapping PD basé sur la notation du crédit, en choisissant la valeur correspondant à l'horizon (1Y, 3Y ou 5Y)
      - LGD converti (exemple "28%" → 0.28)
    
    Les calculs effectués sont :
      - Revenus_Total = Exposure × interest_rate × maturity
      - PD_cumulée = 1 - (1 - PD) ** maturity
      - Expected_Loss = PD_cumulée × LGD × Exposure
      - Profit_Net = Revenus_Total - Expected_Loss
    """
    try:
        # Récupérer l'exposition
        exposure = float(credit["Exposure"])
        # Récupérer la notation (et nettoyer)
        rating = str(credit["Rating"]).strip()
        # Conversion de LGD (si c'est du type "28%", ou un nombre)
        lgd_val = credit["LGD"]
        if isinstance(lgd_val, str):
            lgd_val = float(lgd_val.replace("%", "").strip()) / 100.0
        else:
            lgd_val = float(lgd_val)
        # Pour simplifier, nous supposons que l'EAD est égal à l'Exposure
        ead = exposure
        
        # Sélection de la PD en fonction de l'horizon choisi
        # On suppose que maturity est 1, 3 ou 5 (sinon, on choisit 3 par défaut)
        if rating in pd_mapping:
            pd_values = pd_mapping[rating]
            if abs(maturity - 1) < 0.1:
                pd_value = pd_values.get("1Y", 0.0001)
            elif abs(maturity - 3) < 0.1:
                pd_value = pd_values.get("3Y", 0.0001)
            elif abs(maturity - 5) < 0.1:
                pd_value = pd_values.get("5Y", 0.0001)
            else:
                pd_value = pd_values.get("3Y", 0.0001)
        else:
            pd_value = 0.0001  # Valeur par défaut si la notation n'est pas trouvée
        
        # Calcul du revenu d'intérêts total (intérêt simple)
        interest_revenue = exposure * interest_rate * maturity
        
        # Calcul de la probabilité cumulée de défaut sur la durée
        pd_cum = 1 - (1 - pd_value) ** maturity
        
        # Calcul de la perte attendue (Expected Loss)
        expected_loss = pd_cum * lgd_val * ead
        
        # Calcul du profit net
        net_profit = interest_revenue - expected_loss
        
        return {
            "Exposure": exposure,
            "Rating": rating,
            "LGD": lgd_val,
            "Interest_Rate": interest_rate,
            "Maturity": maturity,
            "Revenus_Total": interest_revenue,
            "PD_annuelle": pd_value,
            "PD_cumulée": pd_cum,
            "Expected_Loss": expected_loss,
            "Profit_Net": net_profit
        }
    except Exception as e:
        st.error(f"Erreur lors de la conversion des données du crédit : {e}")
        return None

# --------------------------
# Configuration de la page Streamlit
# --------------------------
st.set_page_config(page_title="Outil de Tarification RaRoC", layout="wide")
st.title("Outil de Tarification RaRoC - Projet 1")
st.markdown("Application de simulation pour une opération de crédit basée sur le fichier **credit.xlsx**.")

# --------------------------
# Chargement du fichier Excel depuis la barre latérale
# --------------------------
st.sidebar.header("Chargement du fichier Excel")
uploaded_file = st.sidebar.file_uploader("Charger le fichier 'credit.xlsx'", type=["xlsx"])

if uploaded_file is not None:
    data = load_excel(uploaded_file)
    if data is not None:
        # Extraction du mapping PD depuis la feuille "Params"
        pd_mapping = extract_pd_mapping(data)
        
        # Affichage de la feuille Portfolio
        if "Portfolio" in data:
            portfolio_df = data["Portfolio"]
            portfolio_df.columns = portfolio_df.columns.str.strip()  # Nettoyage des noms de colonnes
            st.subheader("Aperçu de la table 'Portfolio' (10 premières lignes)")
            st.dataframe(portfolio_df.head(10))
            
            # Histogramme de la distribution des Exposures
            if "Exposure" in portfolio_df.columns:
                st.subheader("Distribution des Exposures")
                hist = alt.Chart(portfolio_df).mark_bar().encode(
                    alt.X("Exposure:Q", bin=alt.Bin(maxbins=30), title="Exposure (€)"),
                    alt.Y("count()", title="Nombre de crédits"),
                    tooltip=["count()"]
                ).properties(title="Histogramme des Exposures")
                st.altair_chart(hist, use_container_width=True)
        else:
            st.error("La feuille 'Portfolio' n'a pas été trouvée dans le fichier Excel.")
    else:
        st.error("Erreur lors du chargement du fichier.")
else:
    st.info("Veuillez charger le fichier Excel 'credit.xlsx' via la barre latérale.")

# --------------------------
# Formulaire de modification des paramètres de calcul
# --------------------------
st.markdown("## Modification des paramètres de calcul")
# Valeurs par défaut
default_risk_free_rate = 0.02
default_operational_margin = 0.01
default_interest_rate = 0.05    # Taux d'intérêt appliqué pour le calcul de l'opération
default_credit_maturity = 3.0   # Échéance en années (idéalement 1, 3 ou 5)

with st.form("param_form"):
    new_risk_free_rate = st.number_input("Taux sans risque", value=default_risk_free_rate, step=0.001, format="%.3f")
    new_operational_margin = st.number_input("Marge opérationnelle", value=default_operational_margin, step=0.001, format="%.3f")
    new_interest_rate = st.number_input("Taux d'intérêt (annuel)", value=default_interest_rate, step=0.001, format="%.3f")
    new_credit_maturity = st.number_input("Échéance (années)", value=default_credit_maturity, step=0.5)
    submit_params = st.form_submit_button("Enregistrer les paramètres")
    if submit_params:
        risk_free_rate = new_risk_free_rate
        operational_margin = new_operational_margin
        interest_rate = new_interest_rate
        credit_maturity = new_credit_maturity
        st.success("Paramètres mis à jour")
        st.write(f"**Taux sans risque :** {risk_free_rate:.3f} | **Marge opérationnelle :** {operational_margin:.3f} | **Taux d'intérêt :** {interest_rate:.3f} | **Échéance :** {credit_maturity:.1f} ans")
    else:
    # Valeurs par défaut si le formulaire n'est pas soumis
        interest_rate = default_interest_rate
        credit_maturity = default_credit_maturity

# --------------------------
# Formulaire d'analyse d'une opération de crédit
# --------------------------
st.markdown("## Analyse d'une opération de crédit")
with st.form("credit_form"):
    credit_id_input = st.text_input("Saisir l'ID_Credit à analyser (ex : 1, 2, 3, ...)")
    submit_credit = st.form_submit_button("Lancer le calcul")

if submit_credit:
    if uploaded_file is None:
        st.error("Veuillez d'abord charger le fichier Excel.")
    elif "Portfolio" not in data:
        st.error("La feuille 'Portfolio' est introuvable dans le fichier Excel.")
    elif credit_id_input == "":
        st.error("Veuillez saisir un ID_Credit.")
    else:
        try:
            credit_id_int = int(credit_id_input)
        except Exception:
            st.error("L'ID doit être un entier.")
        else:
            # Recherche du crédit dans la feuille Portfolio (colonne "Id")
            credit_data = portfolio_df[portfolio_df["Id"] == credit_id_int]
            if credit_data.empty:
                st.error(f"Aucun crédit trouvé pour l'ID_Credit '{credit_id_input}'.")
            else:
                credit = credit_data.iloc[0]
                result = calculate_credit(credit, interest_rate, credit_maturity, pd_mapping)
                if result is not None:
                    st.markdown("### Mini Compte de Résultat")
                    st.write(f"**ID_Credit :** {credit_id_input}")
                    st.write(f"**Exposure du crédit :** {result['Exposure']:,.2f} €")
                    st.write(f"**Taux d'intérêt appliqué :** {result['Interest_Rate']:.2%}")
                    st.write(f"**Échéance (années) :** {result['Maturity']}")
                    st.write(f"**Revenus d'intérêts total :** {result['Revenus_Total']:,.2f} €")
                    st.write(f"**PD annuelle (pour rating {result['Rating']}) :** {result['PD_annuelle']:.4%}")
                    st.write(f"**Probabilité cumulée sur {result['Maturity']} ans :** {result['PD_cumulée']:.4%}")
                    st.write(f"**LGD :** {result['LGD']:.2%}")
                    st.write(f"**Expected Loss :** {result['Expected_Loss']:,.2f} €")
                    st.write(f"**Profit Net :** {result['Profit_Net']:,.2f} €")
                    
                    # Graphique : Diagramme en barres du mini compte de résultat
                    chart_data = pd.DataFrame({
                        "Indicateur": ["Revenus d'intérêts", "Perte Attendue", "Profit Net"],
                        "Valeur": [result["Revenus_Total"], result["Expected_Loss"], result["Profit_Net"]]
                    })
                    bar_chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X("Indicateur:N", title="Indicateur"),
                        y=alt.Y("Valeur:Q", title="Montant (€)"),
                        tooltip=["Indicateur", "Valeur"]
                    ).properties(title="Graphique du Mini Compte de Résultat")
                    st.altair_chart(bar_chart, use_container_width=True)

# --------------------------
# Remarques et pistes d'amélioration
# --------------------------
st.markdown("""

""")