import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import streamlit as st

# Dati
@st.cache_data
def load_data():
    df_ret = pd.read_csv('returns_monthly.csv', index_col=0, parse_dates=True)
    df_rate = pd.read_csv('rates_monthly.csv', index_col=0, parse_dates=True)
    
    if df_rate.iloc[:, 0].abs().mean() > 0.1:
        df_rate = df_rate / 100
    return df_rate, df_ret

df_rate, df_ret = load_data()



class StressTesting:

    def __init__(self, returns, rates, weights):
        self.returns = returns
        self.rates = rates
        self.weights = np.array(weights)
        self.beta = None
        self.vols = None

    def calculate_betas(self):
        data_joined = pd.concat([self.returns, self.rates], axis=1).dropna()
        periodo_shock = data_joined.loc['2021-01-01':'2024-01-01']
        
        X = periodo_shock.iloc[:, -1].values.reshape(-1, 1) 
        betas = []
        vols = []

        for col in self.returns.columns:
            y = periodo_shock[col].values
            reg = LinearRegression().fit(X, y)
            betas.append(reg.coef_[0])
            vols.append(np.std(y - reg.predict(X)))
            
        self.beta = np.array(betas)
        self.vols = np.array(vols)
        return self.beta, self.vols




class StochasticProcess:

    def __init__(self, betas, volatilities, weights, shock_tassi):
        self.betas = betas
        self.vols = volatilities
        self.weights = weights
        self.shock = shock_tassi
        self.results = []

    def run_simulation(self, n_sim=10000):
        
        for i in range(n_sim):
            
            dW = np.random.normal(0, 1, len(self.betas)) * self.vols          
            # Impatto = (Beta * Shock) + Componente Casuale (dW)
            rendimenti_asset = (self.betas * self.shock) + dW
        
            port_return = np.sum(rendimenti_asset * self.weights)
            self.results.append(port_return)
            
        return np.array(self.results)

#Streamlit

st.title('Stress Test Simulator')
st.markdown("""
Questa applicazione ti permette di valutare come il tuo portafoglio reagirebbe a uno **shock improvviso dei tassi di interesse** (scenario *Hawk* delle banche centrali). 
A differenza delle metriche di rischio standard, questo strumento non guarda solo alla volatilità passata, ma analizza la **sensibilità specifica** di ogni tuo asset rispetto al costo del denaro.

---

### **Come funziona l'analisi?**

* **Stima dei Beta (Regressione):** L'algoritmo analizza il comportamento dei tuoi ETF nel periodo **2020-2023**. Abbiamo scelto questo arco temporale perché rappresenta il "nuovo regime" di mercato, caratterizzato da inflazione persistente e rialzi dei tassi senza precedenti.
* **Motore Monte Carlo:** Attraverso **10.000 simulazioni stocastiche**, il sistema combina l'effetto deterministico dello shock (quanto l'asset dovrebbe scendere in teoria) con la componente casuale e l'incertezza tipica dei mercati finanziari.
* **Metriche di Coda:** Oltre alla perdita media, l'app calcola il **VaR (Value at Risk)** e l'**Expected Shortfall**, indicandoti l'entità delle perdite nei scenari peggiori (il cosiddetto "rischio di coda").



> **Nota Tecnica:** I risultati sono stime statistiche basate su dati mensili. Ricorda che le correlazioni passate non garantiscono performance future, ma forniscono una base solida per una gestione prudente del rischio.
""")

st.divider()

st.info("💡 **Suggerimento:** Inserisci i pesi nella barra laterale e seleziona uno shock (es. +5.0%) per vedere come cambiano il VaR e l'Expected Shortfall.")


select_asset = st.sidebar.multiselect(
    "Scegli gli ETF del tuo portafoglio",
    options=df_ret.columns.tolist()
)

weights = []
if select_asset:
    for asset in select_asset:
        w = st.sidebar.number_input(f'Peso: {asset}', min_value=0.0, max_value=1.0, value=float(1.0/len(select_asset)), step=0.01)
        weights.append(w)
    
    total_w = sum(weights)
    run_allowed = abs(total_w - 1.0) < 0.001
else:
    run_allowed = False

if run_allowed:
    shock_input = st.slider('Shock dei tassi (Punti Percentuali):', 0.0, 10.0, 5.0) / 100

    if st.button('Lancia simulazione'):
        
        # Calcolo Statistico
        st_engine = StressTesting(df_ret[select_asset], df_rate, weights)
        betas, volatilities = st_engine.calculate_betas()
        
        sim_engine = StochasticProcess(betas, volatilities, weights, shock_input)
        results = sim_engine.run_simulation(n_sim=10000)
        
        # PLOT
        st.subheader("Sensibilità (Beta) calcolata nel periodo 2020-2023")
        st.table(pd.DataFrame({'Asset': select_asset, 'Beta': betas}))

        st.divider()
        col1, col2, col3 = st.columns(3)
        
        var_95 = np.percentile(results, 5)
        es_95 = results[results <= var_95].mean()
        
        col1.metric("Rendimento Medio", f"{results.mean():.2%}")
        col2.metric("VaR 95%", f"{var_95:.2%}")
        col3.metric("Expected Shortfall", f"{es_95:.2%}")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(results, bins=100, color='royalblue', alpha=0.7, edgecolor='white')
        ax.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.2%}')
        ax.set_title("Distribuzione Rendimenti Monte Carlo")
        ax.legend()
        st.pyplot(fig)
else:
    st.warning("La somma dei pesi deve essere 100%.")

