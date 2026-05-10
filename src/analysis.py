

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# ============================================================
# FUNKSIONI 1: DATA CLEANING (pastrimi i te dhenave)
# ============================================================

def clean_covid(df):
    """
    Funksion: clean_covid()
    Autori:   Erdona Kadriolli
    
    Qellimi:  Pastron dataset-in RAW dhe shton 4 kolona te reja
              te llogaritura (feature engineering)
    
    Hapat e Pastrimit:
        1. Konverton 'date' ne format datetime
        2. Heq rreshtat me data te zbrazeta
        3. Heq rreshtat duplikate
        4. Pastron whitespace ne emrat
        5. Mbush vlerat NULL me 0
        6. Heq vlerat negative dhe outliers
        7. Sorton sipas dates
    
    Feature Engineering (4 kolona te reja):
        - moving_avg_7d:       Mesatarja levizese 7-ditore
        - cases_per_100k:      Raste per 100,000 banore
        - vaccination_pct:     Perqindja e vaksinimit
        - case_fatality_rate:  Norma e mortalitetit (CFR)
    
    Parametri:
        df (pd.DataFrame): Dataset-i RAW (i pa-rregulluar)
    
    Kthen:
        pd.DataFrame: Dataset-i i pastruar dhe i pasuruar
    """
    
    # Kopjo dataset-in qe te mos modifikohet origjinali
    df = df.copy()
    
    # --------------------------------------------------------
    # HAPI 1: KONVERTIMI I DATES
    # --------------------------------------------------------
    # errors='coerce' kthen NaN per datat e gabuara
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Heq rreshtat me data te zbrazeta (NaN)
    df = df.dropna(subset=['date'])
    
    # --------------------------------------------------------
    # HAPI 2: HEQJA E DUPLIKATEVE
    # --------------------------------------------------------
    df = df.drop_duplicates()
    
    # --------------------------------------------------------
    # HAPI 3: PASTRIMI I WHITESPACE
    # --------------------------------------------------------
    # P.sh.: '  Kosovo  ' -> 'Kosovo'
    if 'location' in df.columns:
        df['location'] = df['location'].str.strip()
    
    # --------------------------------------------------------
    # HAPI 4: TRAJTIMI I VLERAVE NULL
    # --------------------------------------------------------
    # Per kolonat numerike, NULL = 0 (mungesa = pa raste)
    num_cols = ['new_cases', 'total_cases', 'new_deaths',
                'total_deaths', 'new_vaccinations', 'total_vaccinations']
    
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # --------------------------------------------------------
    # HAPI 5: HEQJA E VLERAVE NEGATIVE DHE OUTLIERS
    # --------------------------------------------------------
    # Vlerat negative jane gabime te raportimit
    if 'new_cases' in df.columns:
        df = df[df['new_cases'] >= 0]
        # Outliers: vlera mbi 5000 jane qartesisht gabime (max real ~1450)
        df = df[df['new_cases'] < 5000]
    
    # --------------------------------------------------------
    # HAPI 6: SORTIMI SIPAS DATES
    # --------------------------------------------------------
    df = df.sort_values('date').reset_index(drop=True)
    
    # --------------------------------------------------------
    # HAPI 7: FEATURE ENGINEERING (4 KOLONA TE REJA)
    # --------------------------------------------------------
    
    # Kolona 1: Mesatarja levizese 7-ditore
    # Pse? - Eliminon luhatjet ditore dhe shfaq trendine reale
    df['moving_avg_7d'] = df['new_cases'].rolling(7, min_periods=1).mean().round(1)
    
    # Kolona 2: Raste per 100,000 banore
    # Pse? - Lejon krahasime te drejta mes vendeve me popullsi te ndryshme
    df['cases_per_100k'] = (df['total_cases'] / df['population'] * 100000).round(2)
    
    # Kolona 3: Perqindja e vaksinimit
    # Pse? - Tregon progresin e fushates se vaksinimit
    df['vaccination_pct'] = (df['total_vaccinations'] / df['population'] * 100).clip(0, 100).round(2)
    
    # Kolona 4: Case Fatality Rate (CFR)
    # Pse? - Tregon perqindjen e personave te infektuar qe vdesin
    # np.where() shmang ndarjen me 0 ne fillim te periudhes
    df['case_fatality_rate'] = np.where(
        df['total_cases'] > 0,
        (df['total_deaths'] / df['total_cases'] * 100).round(3),
        0
    )
    
    print(f"[OK] Data cleaning perfunduar: {len(df)} rreshta")
    return df


# ============================================================
# FUNKSIONI 2: STATISTIKAT PERMBLEDHESE
# ============================================================

def get_summary(df):
    """
    Funksion: get_summary()
    Autori:   Erdona Kadriolli
    
    Qellimi:  Llogarit statistikat kryesore te dataset-it
    
    Parametri:
        df (pd.DataFrame): Dataset-i i pastruar
    
    Kthen:
        dict: Dictionary me 8 tregues kryesore
    """
    
    summary = {
        "Rastet Totale":      int(df['total_cases'].max()),
        "Vdekjet Totale":     int(df['total_deaths'].max()),
        "Vaksinime Totale":   int(df['total_vaccinations'].max()),
        "Vaksinimit %":       f"{df['vaccination_pct'].max():.1f}%",
        "Raste Max Ditore":   int(df['new_cases'].max()),
        "Mesatare Ditore":    round(df['new_cases'].mean(), 1),
        "CFR %":              f"{df['case_fatality_rate'].max():.2f}%",
        "Periudha":           f"{df['date'].min().date()} -> {df['date'].max().date()}",
    }
    
    return summary


# ============================================================
# FUNKSIONI 3: IDENTIFIKIMI I VALEVE
# ============================================================

def get_waves(df, multiplier=1.8):
    """
    Funksion: get_waves()
    Autori:   Erdona Kadriolli
    
    Qellimi:  Identifikon valet e pandemise bazuar ne mesataren
              levizese 7-ditore dhe nje pragje dinamike
    
    Logjika:
        - Llogarit mesataren e moving_avg_7d
        - Cdo dite mbi (mesatarja * multiplier) eshte pjese e nje vale
    
    Parametri:
        df (pd.DataFrame): Dataset-i i pastruar
        multiplier (float): Shumezuesi i pragut (default: 1.8)
    
    Kthen:
        pd.DataFrame: Ditet qe jane pjese e valeve
    """
    
    threshold = df['moving_avg_7d'].mean() * multiplier
    waves = df[df['moving_avg_7d'] > threshold][['date', 'new_cases', 'moving_avg_7d']]
    
    print(f"[OK] Valet: {len(waves)} dite mbi pragun {threshold:.0f}")
    return waves


# ============================================================
# FUNKSIONI 4: MATRICA E KORRELACIONIT
# ============================================================

def get_correlation(df):
    """
    Funksion: get_correlation()
    Autori:   Erdona Kadriolli
    
    Qellimi:  Llogarit matricen e korrelacionit Pearson per
              treguesit kryesore te COVID-19
    
    Korrelacioni Pearson:
        +1 = korrelacion i forte pozitiv
         0 = nuk ka korrelacion
        -1 = korrelacion i forte negativ
    
    Parametri:
        df (pd.DataFrame): Dataset-i i pastruar
    
    Kthen:
        pd.DataFrame: Matrica 7x7 e korrelacionit
    """
    
    cols = ['new_cases', 'total_cases', 'new_deaths', 'total_deaths',
            'total_vaccinations', 'vaccination_pct', 'case_fatality_rate']
    
    # .corr() llogarit korrelacionin Pearson automatikisht
    return df[cols].corr()


# ============================================================
# FUNKSIONI 5: PARASHIKIMI ME LINEAR REGRESSION (BONUS)
# ============================================================

def predict_cases(df, days_ahead=30):
    """
    Funksion: predict_cases()
    Autori:   Erdona Kadriolli
    
    Qellimi:  Parashikon rastet totale per ditet e ardhshme duke
              perdorur algoritmin Linear Regression nga scikit-learn
    
    Si Funksionon:
        1. Krijon nje kolone 'days' me numra (0, 1, 2, ..., n)
        2. Trajnon modelin: y = a*x + b (gjen vijen me te mire)
        3. Parashikon vlerat per ditet e ardhshme
        4. Llogarit R² (saktesin e modelit)
    
    Parametri:
        df (pd.DataFrame):  Dataset-i historik
        days_ahead (int):   Sa dite perpara te parashikohen (default: 30)
    
    Kthen:
        dict me celesat:
            - future_dates:  Datat e parashikuara
            - predictions:   Vlerat e parashikuara
            - r_squared:     Saktesia e modelit (0-1)
            - historical_df: Te dhenat historike
    """
    
    # Hapi 1: Pergatit te dhenat - filtro vetem rreshtat me raste > 0
    d = df[df['total_cases'] > 0].copy()
    d['days'] = np.arange(len(d))  # Krijo kolonen 'days': 0, 1, 2, ...
    
    # Hapi 2: Trajno modelin Linear Regression
    model = LinearRegression()
    model.fit(d[['days']], d['total_cases'])
    
    # Hapi 3: Parashiko ditet e ardhshme
    # Krijo array me numrat e diteve te ardhshme: [n+1], [n+2], ...
    future_X = np.array([[len(d) + i] for i in range(1, days_ahead + 1)])
    
    # Aplikoja modelin per parashikim
    predictions = model.predict(future_X)
    
    # Sigurohu qe nuk ka vlera negative (nuk ka kuptim)
    predictions = np.maximum(predictions, 0)
    
    # Hapi 4: Krijo datat e ardhshme
    future_dates = pd.date_range(
        start=d['date'].max() + pd.Timedelta(days=1),
        periods=days_ahead
    )
    
    # Hapi 5: Llogarit R² (saktesia)
    # R² = 1 -> model perfekt
    # R² = 0 -> model i pavlefshëm
    r2 = round(model.score(d[['days']], d['total_cases']), 4)
    
    print(f"[OK] Parashikimi trajnuar: R² = {r2} | {days_ahead} dite perpara")
    
    # Kthe rezultatin si dictionary
    return {
        "future_dates":  future_dates,
        "predictions":   predictions,
        "r_squared":     r2,
        "historical_df": d,
    }


# ============================================================
# TESTIMI I FAJLLIT (ekzekuto: py analysis.py)
# ============================================================

if __name__ == "__main__":
    from load_data import load_covid
    
    print("=" * 55)
    print("  TESTIMI I ANALYSIS.PY")
    print("  Autori: Erdona Kadriolli")
    print("=" * 55)
    
    # Ngarko dhe pastro
    df = load_covid()
    df = clean_covid(df)
    
    # Testimi i statistikave
    print("\n[TEST 1] Statistikat permbledhese:")
    for k, v in get_summary(df).items():
        print(f"  {k:25s}: {v}")
    
    # Testimi i valeve
    print("\n[TEST 2] Valet e pandemise:")
    waves = get_waves(df)
    print(f"  Numri i diteve mbi pragun: {len(waves)}")
    
    # Testimi i korrelacionit
    print("\n[TEST 3] Korrelacioni mes raste-vdekje:")
    corr = get_correlation(df)
    print(f"  R = {corr.loc['total_cases', 'total_deaths']:.3f}")
    
    # Testimi i ML
    print("\n[TEST 4] Parashikimi me Linear Regression:")
    pred = predict_cases(df, days_ahead=30)
    print(f"  R² = {pred['r_squared']}")
    print(f"  Parashikim per 30 dite: {pred['predictions'][0]:.0f} -> {pred['predictions'][-1]:.0f}")
