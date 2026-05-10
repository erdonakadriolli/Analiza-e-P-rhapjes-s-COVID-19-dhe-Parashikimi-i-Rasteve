"""
============================================================
  load_data.py
  
  PERGJEGJESIA: Ngarkimi i te dhenave nga fajllat CSV
  AUTORI:       Erdona Kadriolli
  LENDA:        Shkenca e te Dhenave dhe Vizualizimi me Python
  PROJEKTI:     Analiza e COVID-19 ne Kosove
============================================================

KY FAJLL PERMBAN 2 FUNKSIONE:
  1. load_covid() - ngarkon dataset-in COVID-19 (1,461 rreshta)
  2. load_socio() - ngarkon dataset-in socio-ekonomik (6 rreshta)
"""

import pandas as pd
import os


# ============================================================
# FUNKSIONI 1: NGARKIMI I DATASET-IT COVID-19
# ============================================================

def load_covid(path="../data/clean/covid_kosova_CLEAN.csv"):
    """
    Funksion: load_covid()
    Autori:   Erdona Kadriolli
    
    Qellimi:  Ngarkon dataset-in e pastruar te COVID-19 per Kosoven
    
    Parametri:
        path (str): Rruga drejt fajllit CSV
                    Default: '../data/clean/covid_kosova_CLEAN.csv'
    
    Kthen:
        pd.DataFrame: Dataset-i COVID-19 me 13 kolona
    
    Gabimi:
        FileNotFoundError: Nese fajlli nuk gjendet ne rrugen e dhene
    """
    
    # Hapi 1: Kontrollo nese fajlli ekziston
    if not os.path.exists(path):
        raise FileNotFoundError(f"[GABIM] Fajlli nuk u gjet: {path}")
    
    # Hapi 2: Ngarko dataset-in
    # parse_dates=['date'] konverton automatikisht kolonen 'date' ne datetime
    df = pd.read_csv(path, parse_dates=['date'])
    
    # Hapi 3: Printo informacionin per konfirmim
    print(f"[OK] COVID data ngarkuar: {len(df)} rreshta, {df.shape[1]} kolona")
    
    return df


# ============================================================
# FUNKSIONI 2: NGARKIMI I DATASET-IT SOCIO-EKONOMIK
# ============================================================

def load_socio(path="../data/clean/socioeconomic_kosova_CLEAN.csv"):
    """
    Funksion: load_socio()
    Autori:   Erdona Kadriolli
    
    Qellimi:  Ngarkon dataset-in socio-ekonomik (papunesia, GDP, arsimi)
              Periudha: 2018-2023 (6 vite)
    
    Parametri:
        path (str): Rruga drejt fajllit CSV
                    Default: '../data/clean/socioeconomic_kosova_CLEAN.csv'
    
    Kthen:
        pd.DataFrame: Dataset socio-ekonomik me 9 kolona
                      (year, location, gdp_growth_pct, unemployment_rate,
                       school_close_days, online_learning_pct,
                       poverty_rate_pct, internet_access_pct, period)
    
    Gabimi:
        FileNotFoundError: Nese fajlli nuk gjendet
    """
    
    # Hapi 1: Kontrollo nese fajlli ekziston
    if not os.path.exists(path):
        raise FileNotFoundError(f"[GABIM] Fajlli nuk u gjet: {path}")
    
    # Hapi 2: Ngarko dataset-in
    ds = pd.read_csv(path)
    
    # Hapi 3: Printo informacionin
    print(f"[OK] Socio-economic data ngarkuar: {len(ds)} rreshta, {ds.shape[1]} kolona")
    
    return ds


# ============================================================
# TESTIMI I FAJLLIT (ekzekuto: py load_data.py)
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  TESTIMI I LOAD_DATA.PY")
    print("  Autori: Erdona Kadriolli")
    print("=" * 55)
    
    # Testimi i load_covid()
    print("\n[TEST 1] Ngarkimi i dataset-it COVID-19...")
    df = load_covid()
    print(f"  - Periudha: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  - Kolonat: {list(df.columns)}")
    
    # Testimi i load_socio()
    print("\n[TEST 2] Ngarkimi i dataset-it socio-ekonomik...")
    ds = load_socio()
    print(f"  - Vitet: {ds['year'].tolist()}")
    print(f"  - Kolonat: {list(ds.columns)}")
    
    print("\n[OK] Te dy fajllat u ngarkuan me sukses!")
