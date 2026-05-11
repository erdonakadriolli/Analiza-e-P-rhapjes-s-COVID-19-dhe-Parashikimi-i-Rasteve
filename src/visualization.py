import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import os

sns.set_style("darkgrid")

# =============================
# LOAD DATA
# =============================

def load_data(path="covid_kosova_CLEAN.csv"):
    """Load COVID data with multiple path attempts"""
    # Try multiple possible locations
    possible_paths = [
        path,  # Direct path
        "covid_kosova_CLEAN.csv",  # Same directory
        "../covid_kosova_CLEAN.csv",  # Parent directory
        "../../covid_kosova_CLEAN.csv",  # Two levels up
        os.path.join(os.path.dirname(__file__), "covid_kosova_CLEAN.csv"),  # Same as script
        os.path.join(os.path.dirname(__file__), "../covid_kosova_CLEAN.csv"),  # Parent of script
    ]
    
    for try_path in possible_paths:
        try:
            print(f"Trying to load from: {try_path}")
            df = pd.read_csv(try_path)
            print(f"Successfully loaded from: {try_path}")
            
            # Check if 'date' column exists, if not try 'Date' or other variations
            if 'date' not in df.columns:
                if 'Date' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'])
                elif 'data' in df.columns:
                    df['date'] = pd.to_datetime(df['data'])
                else:
                    # Try to find any date-like column
                    for col in df.columns:
                        if 'date' in col.lower():
                            df['date'] = pd.to_datetime(df[col])
                            break
            else:
                df['date'] = pd.to_datetime(df['date'])
            
            df = df.sort_values('date')
            df.fillna(0, inplace=True)
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading from {try_path}: {e}")
            continue
    
    raise FileNotFoundError(
        f"Could not find covid_kosova_CLEAN.csv in any location.\n"
        f"Tried: {possible_paths}\n"
        f"Current working directory: {os.getcwd()}"
    )


# =============================
# GRAFIKU 1 - RASTET TOTALE
# =============================

def plot_total_cases(df):
    plt.figure(figsize=(14, 6))

    plt.plot(df['date'], df['total_cases'], linewidth=2, color='blue')

    plt.title('Rastet Totale me Kalimin e Kohes', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Rastet Totale')
    plt.xticks(rotation=45)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/01_rastet_totale.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 2 - RASTE DITORE
# =============================

def plot_daily_cases(df):
    df['7_day_avg'] = df['new_cases'].rolling(7).mean()

    plt.figure(figsize=(14, 6))

    plt.bar(df['date'], df['new_cases'], alpha=0.6, color='skyblue', label='Rastet Ditore')
    plt.plot(df['date'], df['7_day_avg'], linewidth=3, color='red', label='Mesatarja 7-Ditore')

    plt.title('Rastet Ditore dhe Mesatarja 7-Ditore', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Rastet')
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/02_rastet_ditore.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 3 - VDEKJET TOTALE
# =============================

def plot_total_deaths(df):
    plt.figure(figsize=(14, 6))

    plt.plot(df['date'], df['total_deaths'], linewidth=3, color='darkred')

    plt.title('Vdekjet Totale', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Vdekjet')
    plt.xticks(rotation=45)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/03_vdekjet_totale.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 4 - VAKSINIMET
# =============================

def plot_vaccinations(df):
    # Check if population column exists
    if 'population' not in df.columns:
        print("Warning: 'population' column not found. Skipping vaccination plot.")
        return
    
    # Avoid division by zero
    if df['population'].sum() == 0:
        print("Warning: Population data is zero. Skipping vaccination plot.")
        return
    
    df['vaccination_percent'] = (df['total_vaccinations'] / df['population']) * 100

    plt.figure(figsize=(14, 6))

    plt.plot(df['date'], df['vaccination_percent'], linewidth=3, color='green')

    plt.title('Perqindja e Vaksinimeve', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Vaksinime %')
    plt.xticks(rotation=45)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/04_vaksinime.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 5 - HISTOGRAM
# =============================

def plot_histogram(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(df['new_cases'], bins=30, kde=True, color='purple')

    plt.title('Histogrami i Rasteve Ditore', fontsize=14, fontweight='bold')
    plt.xlabel('Rastet Ditore')
    plt.ylabel('Frekuenca')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/05_histogram.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 6 - SCATTER PLOT
# =============================

def plot_scatter(df):
    plt.figure(figsize=(10, 6))

    sns.scatterplot(x=df['new_cases'], y=df['new_deaths'], alpha=0.6, color='coral')

    plt.title('Rastet vs Vdekjet', fontsize=14, fontweight='bold')
    plt.xlabel('Rastet e Reja')
    plt.ylabel('Vdekjet e Reja')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/06_scatter_plot.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 7 - PAPUNESIA
# =============================

def plot_unemployment(ds=None):
    periudha = ['Para COVID', 'Gjate COVID', 'Pas COVID']
    papunesia = [24, 31, 19]
    colors = ['green', 'red', 'blue']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(periudha, papunesia, color=colors, alpha=0.7)

    plt.title('Papunesia ne Kosove', fontsize=14, fontweight='bold')
    plt.ylabel('%')
    plt.ylim(0, max(papunesia) * 1.2)

    # Add value labels on bars
    for bar, value in zip(bars, papunesia):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/07_papunesia.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 8 - SHKOLLIMI
# =============================

def plot_education(ds=None):
    labels = ['Shkolla te Hapura', 'Online Learning']
    values = [40, 160]
    colors = ['lightgreen', 'lightcoral']

    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)

    plt.title('Ndikimi ne Arsim', fontsize=14, fontweight='bold')
    plt.axis('equal')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/08_shkollimi.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 9 - GDP
# =============================

def plot_gdp(ds=None):
    vite = ['2019', '2020', '2021', '2022']
    gdp = [4.5, -5.3, 6.1, 3.8]
    colors = ['green' if x >= 0 else 'red' for x in gdp]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(vite, gdp, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.title('GDP Growth', fontsize=14, fontweight='bold')
    plt.ylabel('%')

    # Add value labels
    for bar, value in zip(bars, gdp):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if value >= 0 else -0.3),
                f'{value}%', ha='center', va='bottom' if value >= 0 else 'top', fontweight='bold')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/09_gdp_growth.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 10 - HEATMAP
# =============================

def plot_heatmap(df):
    correlation = df.select_dtypes(include='number').corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)

    plt.title('Heatmap e Korrelacionit', fontsize=14, fontweight='bold')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/10_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 11 - ML
# =============================

def plot_ml_prediction(df):
    df = df.reset_index(drop=True)
    df['day'] = np.arange(len(df))

    X = df[['day']]
    y = df['total_cases']

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    plt.figure(figsize=(14, 6))

    plt.plot(df['date'], y, label='Reale', linewidth=2, color='blue')
    plt.plot(df['date'], predictions, linewidth=3, label='Parashikimi', color='red', linestyle='--')

    plt.legend()
    plt.title('Parashikimi ML me Linear Regression', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Rastet Totale')
    plt.xticks(rotation=45)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/11_parashikimi_ml.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# GRAFIKU 12 - PANELI
# =============================

def plot_dashboard_panel(df):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].plot(df['date'], df['total_cases'], linewidth=2, color='blue')
    axs[0, 0].set_title('Rastet Totale', fontweight='bold')
    axs[0, 0].tick_params(axis='x', rotation=45)

    axs[0, 1].plot(df['date'], df['total_deaths'], linewidth=2, color='darkred')
    axs[0, 1].set_title('Vdekjet Totale', fontweight='bold')
    axs[0, 1].tick_params(axis='x', rotation=45)

    axs[1, 0].plot(df['date'], df['total_vaccinations'], linewidth=2, color='green')
    axs[1, 0].set_title('Vaksinimet', fontweight='bold')
    axs[1, 0].tick_params(axis='x', rotation=45)

    axs[1, 1].plot(df['date'], df['new_cases'], linewidth=2, color='orange')
    axs[1, 1].set_title('Rastet Ditore', fontweight='bold')
    axs[1, 1].tick_params(axis='x', rotation=45)

    plt.suptitle('Paneli Përmbledhës i COVID-19 në Kosovë', fontsize=16, fontweight='bold')
    plt.tight_layout()

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/12_panel_permbledhes.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================
# RUN ALL
# =============================

def generate_all_visualizations():
    try:
        df = load_data()
        print(f"Data loaded successfully! Shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Columns available: {list(df.columns)}")
        
        os.makedirs('outputs', exist_ok=True)
        
        plot_total_cases(df)
        print("✓ Generated: 01_rastet_totale.png")
        
        plot_daily_cases(df)
        print("✓ Generated: 02_rastet_ditore.png")
        
        plot_total_deaths(df)
        print("✓ Generated: 03_vdekjet_totale.png")
        
        plot_vaccinations(df)
        print("✓ Generated: 04_vaksinime.png")
        
        plot_histogram(df)
        print("✓ Generated: 05_histogram.png")
        
        plot_scatter(df)
        print("✓ Generated: 06_scatter_plot.png")
        
        plot_unemployment()
        print("✓ Generated: 07_papunesia.png")
        
        plot_education()
        print("✓ Generated: 08_shkollimi.png")
        
        plot_gdp()
        print("✓ Generated: 09_gdp_growth.png")
        
        plot_heatmap(df)
        print("✓ Generated: 10_heatmap.png")
        
        plot_ml_prediction(df)
        print("✓ Generated: 11_parashikimi_ml.png")
        
        plot_dashboard_panel(df)
        print("✓ Generated: 12_panel_permbledhes.png")
        
        print("\n✅ All visualizations generated successfully in 'outputs' directory!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease make sure the file 'covid_kosova_CLEAN.csv' exists in one of these locations:")
        print("1. Same directory as visualization.py")
        print("2. Parent directory (one level up)")
        print("3. Current working directory")
        print(f"\nCurrent working directory: {os.getcwd()}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

# ============================================================
# ALIASES - per perputhje me main.py
# ============================================================

def plot_deaths(df):
    return plot_total_deaths(df)


def plot_vaccination(df):
    return plot_vaccinations(df)


def plot_prediction(pred):
    historical_df = pred["historical_df"].copy()
    historical_df = historical_df.reset_index(drop=True)

    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(14, 6))

    plt.plot(
        historical_df["date"],
        historical_df["total_cases"],
        label="Te dhenat historike",
        linewidth=2,
        color="blue"
    )

    plt.plot(
        pred["future_dates"],
        pred["predictions"],
        label="Parashikimi",
        linewidth=3,
        color="red",
        linestyle="--"
    )

    plt.title("Parashikimi i Rasteve Totale me Linear Regression", fontsize=14, fontweight="bold")
    plt.xlabel("Data")
    plt.ylabel("Rastet Totale")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/11_parashikimi_ml.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_panel(df, ds=None):
    return plot_dashboard_panel(df)

if __name__ == '__main__':
    generate_all_visualizations()
