import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go



class StressTest:

    def __init__(self, returns, rates, weight, shock_tassi):
        self.returns = returns
        self.rates = rates
        self.weight = np.array(weight)
        self.shock_tassi = shock_tassi


    def calcola_beta(self):
        combined = self.rates.join(self.returns, how='inner').dropna()

        # Periodo
        periodo = combined.loc['2021-01-01':'2023-12-31']

        X = periodo.iloc[:,0].values.reshape(-1, 1)

        self.betas = []
        self.vols = []
        for col in self.returns.columns:
            y = periodo[col].values
            model = LinearRegression().fit(X, y)
            self.betas.append(model.coef_[0])
            self.vols.append(np.std(y - model.predict(X)))
        
        return self.betas, self.vols

    
    def run_simulation(self, n_sim = 1000):
        betas = np.array(self.betas)
        vols = np.array(self.vols)

        betas_ann = betas 
        vols_ann = vols * np.sqrt(12) 

        results = []

        for _ in range(n_sim):
            dw = np.random.standard_t(df=3, size=len(self.betas)) * vols_ann * np.sqrt((3-2)/3)
            
            # Rendimento = Beta*shock + dW
            rendimenti_asset = (betas_ann * (self.shock_tassi/100)) + dw
            port_return = np.sum(rendimenti_asset*self.weight)
            results.append(port_return)

        return np.array(results)*100
    


    def plot_distribution(self, mc_results):
        var_95 = np.percentile(mc_results, 5)

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=mc_results,
            nbinsx=100,
            marker_color='#1f77b4',
            opacity=0.75,
            name='Rendimenti MC'
        ))

        # Linea VaR
        fig.add_vline(x=var_95, line_dash='dash', line_color='red', annotation_text=f'VaR 95%: {var_95:.2f}', annotation_position='top left')

        fig.update_layout(
            title='Distribuzione dei Rendimenti Monte Carlo',
            xaxis_title='Rendimento (%)',
            yaxis_title='Frequenza',
            template='plotly_dark',
            bargap=0.1
        )
        return fig


    
# ----- Data -----
@st.cache_data
def load_data():
    returns = pd.read_csv('returns_monthly.csv', index_col=0)
    rates = pd.read_csv('rates_monthly.csv', index_col=0)
    rates = rates/100
    return returns, rates

returns, rates = load_data()


# ----- Streamlit -----
st.title('Stress Test Simulator')
st.subheader('Metodologia di Analisi')

st.markdown("""
Lo stress test simula l'impatto sul tuo portafoglio di uno shock dei tassi di interesse.
""")


st.markdown("""
**Fase 1: Regressione Lineare** Stimiamo il **Beta** di ogni asset rispetto ai tassi nel periodo 2021-2024 (periodo di forte rialzo dei tassi).
""")
st.latex(r"\Delta Rendimento = \beta \times \Delta Tasso")


st.markdown("""
**Fase 2: Simulazione Stocastica** Utilizziamo il **Moto Browniano Aritmetico** per generare scenari futuri, calcolando:
- **VaR (Value at Risk):** La massima perdita potenziale al 95%.
- **ES (Expected Shortfall):** La media delle perdite oltre il VaR.
    """)

st.markdown("---")


# Sidebar
st.sidebar.header("Configurazione")

asset_list = returns.columns.to_list()
select_asset = st.sidebar.multiselect('Seleziona Asset', asset_list)

pesi = {}
somma_pesi = 0

if select_asset:
    st.sidebar.subheader('Pesi Portafoglio %')
    for a in select_asset:
        pesi[a] = st.sidebar.number_input(f'Peso {a}', min_value=0, max_value=100, value=100//len(select_asset))
    
    somma_pesi = sum(pesi.values())
    if somma_pesi != 100:
        st.sidebar.error(f'La somma dei pesi è {somma_pesi}%, deve essere 100%')

st.sidebar.subheader('Parametri Shock')
shock_tassi = st.sidebar.slider('Shock Tassi (punti %)', 0.0, 10.0, 2.0)
n_sim = st.sidebar.select_slider("Numero Simulazioni", options=[1000, 2000, 5000, 10000], value=1000)


# ----- Esecuzione -----
if st.sidebar.button('ESEGUI STRESS TEST'):
    if not select_asset:
        st.error('Seleziona Almeno un Asset')
    elif somma_pesi != 100:
        st.error('La somma dei pesi deve essere uguale a 100')
    else:
        with st.spinner('Calcolo in corso...'):
            pesi_list = [pesi[a]/100 for a in select_asset]

            regression = StressTest(returns[select_asset], rates.iloc[:,[0]], pesi_list, shock_tassi)
            betas, vols = regression.calcola_beta()

            # Simulazione
            mc = regression.run_simulation(n_sim=n_sim)

            port_beta = np.dot(betas, pesi_list)
            var_95 = np.percentile(mc, 5)
            es = mc[mc <= var_95].mean()

        st.header(f'Risultati Stress Test (Shock: +{shock_tassi}%)')
        m1, m2, m3 = st.columns(3)

        m1.metric("Beta Portafoglio", f"{float(port_beta):.2f}")
        m2.metric("VaR 95%", f'{var_95:.2f}%')
        m3.metric("Expected Shortfall", f'{es:.2f}%')

        st.divider()
        st.plotly_chart(regression.plot_distribution(mc), use_container_width=True)


        st.info("Il modello utilizza una regressione lineare (OLS) per stimare la sensibilità (Beta). Si riconosce che questo approccio approssima la convessità degli asset e assume che le correlazioni passate rimangano stabili durante lo shock.")
        


