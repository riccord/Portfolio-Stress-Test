import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import streamlit as st
from hmmlearn import hmm
from scipy.stats import norm
import plotly.graph_objects as go

# Configurazione Pagina
st.set_page_config(layout="wide")

# --- CARICAMENTO DATI ---
@st.cache_data
def load_data():
    # Carichiamo entrambi i dataset
    df_ret_monthly = pd.read_csv('returns_monthly.csv', index_col=0, parse_dates=True)
    df_ret_daily = pd.read_csv('returns_daily.csv', index_col=0, parse_dates=True)
    df_rate = pd.read_csv('rates_monthly.csv', index_col=0, parse_dates=True)
    
    if df_rate.iloc[:, 0].abs().mean() > 0.1:
        df_rate = df_rate / 100
        
    return df_rate, df_ret_monthly, df_ret_daily

df_rate, df_ret_monthly, df_ret_daily = load_data()

# --- CLASSI ---

class StressTesting:
    def __init__(self, returns, rates, weights):
        self.returns = returns
        self.rates = rates
        self.weights = np.array(weights)
        self.beta = None
        self.vols = None

    def calculate_betas(self):
        # 1. Normalizziamo gli indici temporali per essere sicuri che coincidano (solo Anno-Mese)
        returns_tmp = self.returns.copy()
        rates_tmp = self.rates.copy()
        
        returns_tmp.index = returns_tmp.index.to_period('M')
        rates_tmp.index = rates_tmp.index.to_period('M')

        # 2. Uniamo i dati basandoci sul periodo mensile
        data_joined = pd.concat([returns_tmp, rates_tmp], axis=1).dropna()
        
        # 3. Filtriamo per le date richieste (usando stringhe di periodo)
        try:
            periodo_shock = data_joined.loc['2021-01':'2024-01']
        except KeyError:
            # Se le date specifiche falliscono, prendiamo l'intersezione disponibile
            periodo_shock = data_joined

        if periodo_shock.empty:
            st.error("ERRORE CRITICO: Dopo l'unione dei file, il dataset è vuoto. Controlla che le date in 'rates_monthly.csv' arrivino almeno al 2021.")
            return None, None
        
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
            rendimenti_asset = (self.betas * self.shock) + dW
            port_return = np.sum(rendimenti_asset * self.weights)
            self.results.append(port_return)
        return np.array(self.results)

class risk_spectral:
    def __init__(self, df):
        self.df = df
        self.X = df.values.reshape(-1, 1)
        self.model = None
        self.prob = None
        self.means_ = None
        self.covars_ = None

    def train_hmm(self):
        model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=500, random_state=100)
        model.fit(self.X)
        
        idx = np.argsort(model.covars_.flatten())
        
        self.model = model
        self.means_ = model.means_[idx]
        self.get_covars = model.covars_[idx]
        
        raw_probs = model.predict_proba(self.X)
        self.prob = raw_probs[:, idx]

    def get_metrics(self, index, alpha=0.05):
        p = self.prob[index]
        x_grid = np.linspace(self.X.min() - 0.1, self.X.max() + 0.1, 1000)
        pdf_mixture = np.zeros_like(x_grid)
        
        for i in range(3):
            mu = self.means_[i][0]
            sigma = np.sqrt(self.get_covars[i][0][0])
            pdf_mixture += p[i] * norm.pdf(x_grid, mu, sigma)
        
        dx = x_grid[1] - x_grid[0]
        cdf_mixture = np.cumsum(pdf_mixture) * dx
        
        try:
            idx_var = np.where(cdf_mixture >= alpha)[0][0]
            var_val = x_grid[idx_var]
            es_val = np.sum(x_grid[:idx_var] * pdf_mixture[:idx_var] * dx) / alpha
        except:
            var_val, es_val = 0, 0
            
        return var_val, es_val, x_grid, pdf_mixture

    def plot_spectral_risk(self, index):
        var_val, es_val, x_grid, pdf_mixture = self.get_metrics(index)
        current_return = self.X[index][0]
        p = self.prob[index]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_grid, pdf_mixture, color='black', lw=3, label='Spectral Risk Distribution (f_tot)')
        ax.fill_between(x_grid, pdf_mixture, alpha=0.1, color='black')
        
        colors = ['green', 'orange', 'red']
        labels = ['Calmo', 'Intermedio', 'Crisi']
        for i in range(3):
            mu = self.means_[i][0]
            sigma = np.sqrt(self.get_covars[i][0][0])
            ax.plot(x_grid, p[i] * norm.pdf(x_grid, mu, sigma), '--', color=colors[i], alpha=0.6, 
                    label=f'{labels[i]} (prob: {p[i]:.1%})')

        ax.axvline(var_val, color='darkorange', linestyle='-', lw=2, label=f'VaR 95%: {var_val:.2%}')
        ax.axvline(es_val, color='red', linestyle='-', lw=2, label=f'ES 95%: {es_val:.2%}')
        ax.axvline(current_return, color='blue', lw=3, label=f'Rendimento Attuale: {current_return:.2%}')
        
        ax.set_title(f"Analisi Spettrale al {self.df.index[index].date()}", fontsize=14)
        ax.set_xlabel("Rendimento")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(alpha=0.3)
        return fig
    
    def plot_3d_interattivo(self):
        z_regimes = np.argmax(self.prob, axis=1)
        df_p = pd.DataFrame({'Tempo': np.arange(len(self.X)), 'Rendimenti': self.X.flatten(), 'Regime': z_regimes})
        fig = go.Figure()
        colors = {0: 'navy', 1: 'lime', 2: 'crimson'}
        names = {0: 'Calmo', 1: 'Intermedio', 2: 'Crisi'}
        for i in range(3):
            curr = df_p[df_p['Regime'] == i]
            fig.add_trace(go.Scatter3d(x=curr['Tempo'], y=curr['Rendimenti'], z=curr['Regime'], 
                                       mode='markers', marker=dict(size=3, color=colors[i], opacity=0.7), name=names[i]))
        fig.update_layout(scene=dict(xaxis_title='Tempo', yaxis_title='Rendimenti', zaxis_title='Regime'), 
                          margin=dict(l=0, r=0, b=0, t=0), height=600)
        return fig

# --- SIDEBAR PER ASSET E PESI ---
st.sidebar.header("Selezione Portafoglio")
select_asset = st.sidebar.multiselect(
    "Scegli gli ETF del tuo portafoglio",
    options=df_ret_monthly.columns.tolist()
)

weights = []
if select_asset:
    for asset in select_asset:
        w = st.sidebar.number_input(f'Peso: {asset}', 0.0, 1.0, 1.0/len(select_asset), 0.05)
        weights.append(w)
    
    total_w = sum(weights)
    run_allowed = abs(total_w - 1.0) < 0.001
    if not run_allowed:
        st.sidebar.error(f"La somma dei pesi deve essere 100% (Attuale: {total_w:.1%})")
else:
    run_allowed = False

# --- NAVIGAZIONE TAB ---
tab1, tab2 = st.tabs(['Stress Test Portfolio', 'Spectral Risk'])

# PAGINA 1
with tab1:
    st.title('Stress Test Simulator')
    st.markdown("""
    Questa applicazione ti permette di valutare come il tuo portafoglio reagirebbe a uno **shock improvviso dei tassi di interesse** (scenario *Hawk* delle banche centrali). 
    A differenza delle metriche di rischio standard, questo strumento non guarda solo alla volatilità passata, ma analizza la **sensibilità specifica** di ogni tuo asset rispetto al costo del denaro.

    ---

    ### **Come funziona l'analisi?**

    * **Stima dei Beta (Regressione):** L'algoritmo analizza il comportamento dei tuoi ETF nel periodo **2020-2023**.Ho scelto questo arco temporale perché rappresenta il "nuovo regime" di mercato, caratterizzato da inflazione persistente e rialzi dei tassi senza precedenti.
    * **Motore Monte Carlo:** Attraverso **10.000 simulazioni stocastiche**, il sistema combina l'effetto deterministico dello shock (quanto l'asset dovrebbe scendere in teoria) con la componente casuale e l'incertezza tipica dei mercati finanziari.
    * **Metriche di Coda:** Oltre alla perdita media, l'app calcola il **VaR (Value at Risk)** e l'**Expected Shortfall**, indicandoti l'entità delle perdite nei scenari peggiori (il cosiddetto "rischio di coda").

    > **Nota Tecnica:** I risultati sono stime statistiche basate su dati mensili. Ricorda che le correlazioni passate non garantiscono performance future, ma forniscono una base solida per una gestione prudente del rischio.
    """)

    st.divider()
    st.info("💡 **Suggerimento:** Inserisci i pesi nella barra laterale e seleziona uno shock (es. +5.0%) per vedere come cambiano il VaR e l'Expected Shortfall.")

    if run_allowed:
        shock_input = st.slider('Shock dei tassi (Punti Percentuali):', 0.0, 10.0, 5.0) / 100
        if st.button('Lancia simulazione'):
            st_engine = StressTesting(df_ret_monthly[select_asset], df_rate, weights)
            betas, volatilities = st_engine.calculate_betas()
            sim_engine = StochasticProcess(betas, volatilities, weights, shock_input)
            results = sim_engine.run_simulation(n_sim=10000)

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
        st.warning("Configura il portafoglio nella sidebar (somma pesi = 100%) per procedere.")

# PAGINA 2
with tab2:
    st.header("Analisi del Rischio Spettrale (HMM)")
    st.markdown("""
    Questa sezione analizza la **natura mutevole del rischio**. Invece di una campana fissa, il modello 
    **Hidden Markov Model (HMM)** identifica 3 regimi di mercato (Calmo, Turbolento, Crisi) 
    e adatta la stima del rischio in base alla probabilità di trovarsi in uno di essi oggi.
    """)

    st.markdown("""
    ### **Logica del Modello di Rischio Spettrale**

    Individuati i **3 stati**, per ogni stato $k$ definiamo l'intera funzione di densità di probabilità $f_k(r)$. In ogni giorno l'HMM ci dirà quali sono le probabilità di trovarci in uno stadio $\pi_{k,t}$.

    Per calcolare le metriche di rischio non scegliamo una sola distribuzione di uno stato, ma **fondiamo le 3 distribuzioni in una sola**, pesandole per la probabilità attuale:

    $$f_{tot}(r) = \pi_{bull,t} \cdot f_{bull}(r) + \pi_{mid,t} \cdot f_{mid}(r) + \pi_{bear,t} \cdot f_{bear}(r)$$

    In questo modo otteniamo la nostra **Spectral Risk Distribution**. Si riesce a creare, quindi, una distribuzione **"multiforma"**, che cambia profilo ogni giorno in base alla probabilità tra i diversi regimi di mercato.
    """)

    if select_asset and run_allowed:
        port_daily = (df_ret_daily[select_asset] * weights).sum(axis=1)

        @st.cache_resource
        def init_spectral_model(_data_series):
            tmp_df = pd.DataFrame(_data_series, columns=['returns'])
            model = risk_spectral(tmp_df)
            model.train_hmm()
            return model

        spectral_engine = init_spectral_model(port_daily)

        st.divider()
        idx = st.select_slider(
            "Seleziona una data storica per vedere la distribuzione del rischio in quel momento:",
            options=range(len(port_daily)),
            format_func=lambda x: port_daily.index[x].strftime('%Y-%m-%d'),
            value=len(port_daily)-1
        )

        v_val, e_val, _, _ = spectral_engine.get_metrics(idx)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Rendimento Giorno", f"{port_daily.iloc[idx]:.2%}")
        c2.metric("VaR 95% Spettrale", f"{v_val:.2%}")
        c3.metric("Expected Shortfall", f"{e_val:.2%}")

        fig_spectral = spectral_engine.plot_spectral_risk(idx)
        st.pyplot(fig_spectral)
        st.info("💡 **Nota:** La campana cambia profilo ogni giorno in base alla probabilità tra i diversi regimi di mercato.")

        st.divider()
        st.subheader("Mappa Tridimensionale dei Regimi")
        st.markdown("""
        Questo grafico mostra come il portafoglio si sposta tra i diversi 'piani' di rischio nel tempo. 
        I punti sono colorati in base al regime dominante identificato dall'HMM.
        """)
        
        fig_3d = spectral_engine.plot_3d_interattivo()
        st.plotly_chart(fig_3d, use_container_width=True)

    else:
        st.warning("Seleziona gli asset e imposta i pesi (100%) nella barra laterale per attivare l'analisi spettrale.")