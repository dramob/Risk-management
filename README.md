# Risk-management


## 📁 Structure du dépôt
├── Projet1_RAROC
│   ├── Raroc.py
│   └── credit.xlsx
│
├── Projet2_MonteCarlo_Portefeuille_CDO
│   ├── montecarlo.py
│   └── credit.xlsx
│
├── Projet3_Pricing_Options_MonteCarlo
│   └── option_pricing_mc.py
│
├── Projet4_BlackScholes_CIR_Merton
│   ├── black_scholes_results.csv
│   ├── cir_results.csv
│   ├── merton_results.csv
│   ├── black_scholes_plot.png
│   ├── cir_plot.png
│   ├── merton_plot.png
│   └── models.py
---

## 🚀 Projets réalisés

### 🔹 Projet 1 : RAROC (Risk-Adjusted Return On Capital)
- Application Streamlit permettant la tarification et l'analyse d'une opération de crédit basée sur les indicateurs RAROC.

**Technologies utilisées** :
- Python (Pandas, Altair)
- Streamlit

**Script** : [`Raroc.py`](Projet1_RAROC/Raroc.py)

---

### 🔹 Projet 2 : Gestion des risques du portefeuille et CDO (Monte Carlo)
- Simulation des pertes d'un portefeuille de crédits à l'aide d'un modèle gaussien à deux facteurs.
- Calcul des indicateurs de risque (Expected Loss, VaR, ES) et pertes sur tranches de CDO.

**Technologies utilisées** :
- Python (NumPy, SciPy, Altair, Streamlit)

**Script** : [`montecarlo.py`](Projet2_MonteCarlo_Portefeuille_CDO/montecarlo.py)

---

### 🔹 Projet 3 : Pricing d'Options par Monte Carlo
- Simulation Monte Carlo pour valoriser différentes options financières : Vanilla Call/Put, Tunnel, Himalaya, Napoléon.

**Technologies utilisées** :
- Python (NumPy, Pandas, Altair, SciPy, Streamlit)

**Script** : [`option_pricing_mc.py`](Projet3_Pricing_Options_MonteCarlo/option_pricing_mc.py)

---

### 🔹 Projet 4 : Modèles financiers (Black-Scholes, CIR, Merton)
- Résolution numérique des équations différentielles partielles avec le schéma de Crank-Nicolson.
- Implémentation des modèles :
  - **Black-Scholes** (options européennes)
  - **CIR** (taux d'intérêt)
  - **Merton** (pricing d'options avec sauts)

**Technologies utilisées** :
- Python (NumPy, Matplotlib, Pandas)

**Script** : [`models.py`](Projet4_BlackScholes_CIR_Merton/models.py)

---

## 📊 Résultats numériques (Projet 4)
Les résultats numériques générés par les modèles du projet 4 sont disponibles sous format CSV :
- [Résultats Black-Scholes](Projet4_BlackScholes_CIR_Merton/black_scholes_results.csv)
- [Résultats CIR](Projet4_BlackScholes_CIR_Merton/cir_results.csv)
- [Résultats Merton](Projet4_BlackScholes_CIR_Merton/merton_results.csv)

Les graphiques associés sont également disponibles :
- Black-Scholes : ![Black-Scholes](Projet4_BlackScholes_CIR_Merton/black_scholes_plot.png)
- CIR : ![CIR](Projet4_BlackScholes_CIR_Merton/cir_plot.png)
- Merton : ![Merton](Projet4_BlackScholes_CIR_Merton/merton_plot.png)

---


## ⚙️ Installation et Utilisation
Pour exécuter localement ces projets :

**Cloner le dépôt** :
```bash
git clone https://github.com/alia-drame/risk-management.git
cd risk-management
