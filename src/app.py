import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Page configuration - more compact
st.set_page_config(
    page_title='COVID Dashboard Kosovo',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Custom CSS for compact display
st.markdown("""
<style>
    .main > div {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 14px;
    }
    /* Reduce spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        margin-top: 0rem;
        margin-bottom: 0.5rem;
        font-size: 2rem;
    }
    h3 {
        margin-top: 0rem;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD DATA WITH CACHING
# =============================

@st.cache_data
def load_data():
    """Load COVID data from multiple possible locations"""
    possible_paths = [
        "covid_kosova_CLEAN.csv",
        "../covid_kosova_CLEAN.csv",
        "../../covid_kosova_CLEAN.csv",
        os.path.join(os.path.dirname(__file__), "covid_kosova_CLEAN.csv"),
        os.path.join(os.path.dirname(__file__), "../covid_kosova_CLEAN.csv"),
    ]
    
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            
            # Handle date column variations
            if 'date' not in df.columns:
                if 'Date' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'])
                elif 'data' in df.columns:
                    df['date'] = pd.to_datetime(df['data'])
                else:
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
            st.warning(f"Error loading from {path}: {e}")
            continue
    
    st.error("Could not find covid_kosova_CLEAN.csv file!")
    return pd.DataFrame()

# Load data
df = load_data()
if df.empty:
    st.stop()

# Title - compact
st.title('📊 COVID-19 Kosovo Dashboard')

# =============================
# METRIC CARDS - More compact
# =============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('🦠 Total Cases', f"{int(df['total_cases'].max()):,}")

with col2:
    st.metric('💀 Total Deaths', f"{int(df['total_deaths'].max()):,}")

with col3:
    st.metric('💉 Total Vaccinations', f"{int(df['total_vaccinations'].max()):,}")

with col4:
    if 'population' in df.columns and df['population'].iloc[0] > 0:
        vax_rate = (df['total_vaccinations'].max() / df['population'].iloc[0]) * 100
        st.metric('📊 Vaccination Rate', f"{vax_rate:.1f}%")

st.markdown("---")

# =============================
# TABS - Compact version
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '📈 Trends', '💼 Employment', '📚 Education', '🔗 Correlation', '🤖 ML Prediction'
])

# =============================
# TAB 1 - TRENDS (Combined)
# =============================
with tab1:
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Total Cases')
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df['date'], df['total_cases'], linewidth=2, color='blue')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Total Cases', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader('Total Deaths')
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df['date'], df['total_deaths'], linewidth=2, color='darkred')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Total Deaths', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Daily cases with 7-day average
    st.subheader('Daily Cases (7-day Average)')
    df['7_day_avg'] = df['new_cases'].rolling(7).mean()
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df['date'], df['new_cases'], alpha=0.5, color='skyblue', label='Daily Cases', width=0.8)
    ax.plot(df['date'], df['7_day_avg'], linewidth=2, color='red', label='7-day Average')
    ax.set_xlabel('Date', fontsize=9)
    ax.set_ylabel('Cases', fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Vaccinations if available
    if 'population' in df.columns and df['population'].sum() > 0:
        st.subheader('Vaccination Progress')
        vaccination_percent = (df['total_vaccinations'] / df['population'].replace(0, np.nan)) * 100
        vaccination_percent.fillna(0, inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['date'], vaccination_percent, linewidth=2, color='green')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Vaccination %', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =============================
# TAB 2 - EMPLOYMENT
# =============================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Unemployment Rate')
        periudha = ['Pre-COVID', 'During COVID', 'Post-COVID']
        papunesia = [24, 31, 19]
        colors = ['green', 'red', 'blue']
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(periudha, papunesia, color=colors, alpha=0.7)
        ax.set_ylabel('%', fontsize=9)
        ax.set_ylim(0, max(papunesia) * 1.2)
        
        for bar, value in zip(bars, papunesia):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{value}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader('GDP Growth')
        vite = ['2019', '2020', '2021', '2022']
        gdp = [4.5, -5.3, 6.1, 3.8]
        colors = ['green' if x >= 0 else 'red' for x in gdp]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(vite, gdp, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('GDP Growth %', fontsize=9)
        
        for bar, value in zip(bars, gdp):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if value >= 0 else -0.5),
                   f'{value}%', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Statistics explanation
    st.info("📈 **Impact:** Unemployment increased to 31% during pandemic, then dropped to 19% post-pandemic. GDP showed recovery in 2021-2022.")

# =============================
# TAB 3 - EDUCATION
# =============================
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Education Impact')
        labels = ['Open Schools', 'Online Learning']
        values = [40, 160]
        colors = ['lightgreen', 'lightcoral']
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Learning Methods During Pandemic', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader('Key Statistics')
        st.metric("Students Affected", "200,000+")
        st.metric("Schools Closed", "100%", delta="During Peak")
        st.metric("Digital Divide", "40%", delta="Lack of Internet Access")
    
    st.info("📚 **Finding:** 160,000 students switched to online learning, while only 40,000 continued normal schooling, highlighting digital infrastructure challenges.")

# =============================
# TAB 4 - CORRELATION (Compact)
# =============================
with tab4:
    st.subheader('Feature Correlation Heatmap')
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    if not numeric_df.empty:
        # Limit to key columns for better visibility
        key_cols = [col for col in ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
                                     'total_vaccinations', 'new_vaccinations'] 
                   if col in numeric_df.columns]
        
        if key_cols:
            correlation = numeric_df[key_cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True, ax=ax, annot_kws={'size': 8})
            ax.set_title('Correlation Matrix', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.info("🔍 **Interpretation:** Values close to 1 = strong positive correlation, -1 = strong negative correlation, 0 = no correlation.")
        else:
            st.warning("Not enough numeric columns for correlation analysis")
    else:
        st.warning("No numeric data available")

# =============================
# TAB 5 - ML PREDICTION (Compact)
# =============================
with tab5:
    st.subheader('COVID-19 Case Prediction')
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        days = st.slider('Prediction Days', 7, 180, 30, key='compact_slider')
        predict_button = st.button('🚀 Generate Prediction', type='primary', use_container_width=True)
    
    with col2:
        st.write("**Model:** Linear Regression")
        st.write("**Features:** Time-based (days since start)")
    
    if predict_button:
        with st.spinner('Generating prediction...'):
            # Create a copy to avoid modifying original dataframe
            df_copy = df.reset_index(drop=True)
            df_copy['day'] = np.arange(len(df_copy))
            
            X = df_copy[['day']]
            y = df_copy['total_cases']
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_days = np.arange(len(df_copy) + days).reshape(-1, 1)
            predictions = model.predict(future_days)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Plot historical data (last 100 days for better visibility)
            history_len = min(100, len(df_copy))
            ax.plot(range(history_len), df_copy['total_cases'].iloc[-history_len:], 
                   label='Historical', linewidth=2, color='blue')
            
            # Plot all predictions
            ax.plot(range(len(df_copy) + days), predictions, 
                   label='Prediction', linewidth=2, color='red', linestyle='--')
            
            # Add vertical line
            ax.axvline(x=len(df_copy)-1, color='green', linestyle=':', 
                      label='Prediction Start', linewidth=1.5)
            
            ax.set_title(f'{days}-Day Prediction', fontweight='bold', fontsize=10)
            ax.set_xlabel('Days', fontsize=9)
            ax.set_ylabel('Total Cases', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Show future predictions
            future_predictions = predictions[-days:]
            final_prediction = int(future_predictions[-1])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction Period", f"{days} days")
            with col2:
                st.metric("Final Prediction", f"{final_prediction:,}")
            with col3:
                growth = ((final_prediction - df_copy['total_cases'].iloc[-1]) / df_copy['total_cases'].iloc[-1]) * 100
                st.metric("Expected Growth", f"{growth:.1f}%")
            
            st.success(f'✅ Prediction generated successfully! Model predicts {final_prediction:,} total cases after {days} days.')
            
            # Model metrics in expander
            with st.expander("📊 Model Details"):
                r_squared = model.score(X, y)
                st.write(f"**R-squared:** {r_squared:.4f}")
                st.write(f"**Daily Growth Rate:** {model.coef_[0]:.2f} cases/day")
                st.write(f"**Intercept:** {model.intercept_:.2f}")
                
                if r_squared > 0.8:
                    st.success("✅ Model performance: HIGH (R² > 0.8)")
                elif r_squared > 0.6:
                    st.info("📊 Model performance: MEDIUM (0.6 < R² < 0.8)")
                else:
                    st.warning("⚠️ Model performance: LOW (R² < 0.6)")

# Footer - compact
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>COVID-19 Kosovo Dashboard | Data Analytics Project</p>
    </div>
    """,
    unsafe_allow_html=True
)
