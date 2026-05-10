"""
============================================================
  main.py
  
  PERGJEGJESIA: Fajlli kryesor qe ekzekuton gjithcka
  AUTORI:       Erdona Kadriolli
  LENDA:        Shkenca e te Dhenave dhe Vizualizimi me Python
  PROJEKTI:     Analiza e COVID-19 ne Kosove
============================================================

KY FAJLL:
  - Eshte 'orkestratori' i projektit
  - Therret te gjitha funksionet nga load_data.py, analysis.py
  - Therret funksionet grafike nga visualization.py (Fatlumi)
  - Printon rezultatet ne ekran

EKZEKUTIMI:
  cd src
  py main.py
"""

import sys
import os

# Shto folderin aktual ne path qe te gjenden modulet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importo funksionet e mia (Erdona)
from load_data import load_covid, load_socio
from analysis import (
    clean_covid,
    get_summary,
    get_waves,
    get_correlation,
    predict_cases,
)

# Importo funksionet grafike (puna e Fatlumit)
from visualization import (
    plot_total_cases,
    plot_daily_cases,
    plot_deaths,
    plot_vaccination,
    plot_histogram,
    plot_scatter,
    plot_unemployment,
    plot_education,
    plot_gdp,
    plot_heatmap,
    plot_prediction,
    plot_panel,
)


# ============================================================
# KONFIGURIMI
# ============================================================

COVID_PATH = "../data/clean/covid_kosova_CLEAN.csv"
SOCIO_PATH = "../data/clean/socioeconomic_kosova_CLEAN.csv"
PREDICT_DAYS = 30


# ============================================================
# WORKFLOW KRYESOR
# ============================================================

def main():
    """
    Funksioni kryesor qe ekzekuton te gjitha hapat ne rendin e duhur:
    
        1. Ngarkimi i te dhenave
        2. Pastrimi i te dhenave (data cleaning)
        3. Analiza statistikore
        4. Identifikimi i valeve te pandemise
        5. Llogaritja e korrelacionit
        6. Krijimi i 12 grafikave
        7. Parashikimi me Machine Learning (BONUS)
    """
    
    # ========================================================
    # HEADER
    # ========================================================
    print("=" * 60)
    print("  COVID-19 KOSOVA - Analiza e Plote")
    print("  Autori: Erdona Kadriolli")
    print("  Lenda: Shkenca e te Dhenave dhe Vizualizimi me Python")
    print("=" * 60)
    
    # ========================================================
    # HAPI 1: NGARKIMI I TE DHENAVE
    # ========================================================
    print("\n[1] Ngarkimi i te dhenave...")
    df = load_covid(COVID_PATH)
    ds = load_socio(SOCIO_PATH)
    
    # ========================================================
    # HAPI 2: PASTRIMI (DATA CLEANING)
    # ========================================================
    print("\n[2] Pastrimi i te dhenave...")
    df = clean_covid(df)
    
    # ========================================================
    # HAPI 3: ANALIZA STATISTIKORE
    # ========================================================
    print("\n[3] Analiza statistikore...")
    summary = get_summary(df)
    print("\n  --- PERMBLEDHJA COVID - KOSOVA ---")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")
    
    # ========================================================
    # HAPI 4: IDENTIFIKIMI I VALEVE
    # ========================================================
    print("\n[4] Identifikimi i valeve te pandemise...")
    waves = get_waves(df, multiplier=1.8)
    print(f"  Ditet ne periudha kritike: {len(waves)}")
    
    # ========================================================
    # HAPI 5: MATRICA E KORRELACIONIT
    # ========================================================
    print("\n[5] Llogaritja e korrelacionit...")
    corr = get_correlation(df)
    r = corr.loc['total_cases', 'total_deaths']
    print(f"  Korrelacioni Raste-Vdekje: R = {r:.3f}")
    
    # ========================================================
    # HAPI 6: VIZUALIZIMET (12 GRAFIKE)
    # ========================================================
    print("\n[6] Krijimi i 12 grafikave...")
    
    # Grafiket epidemiologjike (4)
    plot_total_cases(df)        # 01 - Rastet totale
    plot_daily_cases(df)        # 02 - Raste ditore + 7d avg
    plot_deaths(df)             # 03 - Vdekjet totale
    plot_vaccination(df)        # 04 - Vaksinime %
    
    # Grafiket statistikore (3)
    plot_histogram(df)          # 05 - Histogram
    plot_scatter(df)            # 06 - Scatter raste vs vdekje
    plot_heatmap(corr)          # 10 - Heatmap korrelacioni
    
    # Grafiket socio-ekonomike (3)
    plot_unemployment(ds)       # 07 - Papunesia
    plot_education(ds)          # 08 - Shkollimi
    plot_gdp(ds)                # 09 - GDP growth
    
    # Panel permbledhes
    plot_panel(df, ds)          # 12 - Panel 4-in-1
    
    # ========================================================
    # HAPI 7: PARASHIKIMI ME ML (BONUS)
    # ========================================================
    print(f"\n[7] BONUS: Parashikimi me Linear Regression ({PREDICT_DAYS} dite)...")
    pred = predict_cases(df, days_ahead=PREDICT_DAYS)
    plot_prediction(pred)       # 11 - Grafiku i parashikimit
    
    print(f"\n  Saktesia e modelit: R² = {pred['r_squared']}")
    print(f"  Vlera e fillimit: {pred['predictions'][0]:.0f}")
    print(f"  Vlera e fundit:   {pred['predictions'][-1]:.0f}")
    
    # ========================================================
    # PERFUNDIMI
    # ========================================================
    print("\n" + "=" * 60)
    print("  PROJEKTI PERFUNDOI ME SUKSES!")
    print("  Grafikat u ruajten ne: ../outputs/")
    print("  Numri total i grafikave: 12")
    print("=" * 60)


# ============================================================
# EKZEKUTIMI
# ============================================================

if __name__ == "__main__":
    main()
