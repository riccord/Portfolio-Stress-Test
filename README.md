# Portfolio-Stress-Test
Questo progetto è un'applicazione interattiva sviluppata in Streamlit che permette di valutare la resilienza di un portafoglio finanziario a fronte di shock macroeconomici, specificamente variazioni dei tassi di interesse.
Il simulatore segue un processo quantitativo diviso in tre fasi:Stima della Sensibilità (Beta): 
- Utilizza una regressione lineare (OLS) per calcolare il coefficiente $\beta$ di ogni asset rispetto ai tassi storici (periodo 2021-2023). Questo indica quanto      l'asset è storicamente sensibile ai movimenti del costo del denaro.
- Simulazione Monte Carlo: Viene generata una distribuzione di 1000+ scenari futuri utilizzando un modello stocastico dove il rendimento dell'asset è definito        come:
  $$R_p = (\beta \times \Delta Tasso) + \epsilon$$; dove $\epsilon$ rappresenta il rischio specifico dell'asset (volatilità non legata ai tassi)


Puoi provare il simulatore interattivo qui: [https://portfolio-stress-test-9adgtanke2bxlp9tk2a7b5.streamlit.app/]
