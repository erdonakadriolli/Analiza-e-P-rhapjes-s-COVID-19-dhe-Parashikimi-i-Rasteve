# COVID-19 Data Analysis and Visualization

## 📌 Project Overview

This project was developed as part of the course **"Shkenca e të Dhënave dhe Vizualizimi me Python"**.

The main objective of this project is to analyze COVID-19 data for Kosovo and present the results through statistical analysis, data visualization, and basic predictive modeling. The project focuses on understanding the spread of COVID-19, analyzing total cases, deaths, vaccination progress, socio-economic effects, and predicting future case trends using Linear Regression.

The project includes data cleaning, feature engineering, statistical analysis, visualizations, and an interactive dashboard built with Streamlit.

---

## 📁 Dataset

The COVID-19 dataset is based on data from **Our World in Data** and is filtered for Kosovo.

The dataset includes columns such as:

- `date`
- `location`
- `population`
- `new_cases`
- `total_cases`
- `new_deaths`
- `total_deaths`
- `new_vaccinations`
- `total_vaccinations`
- `moving_avg_7d`
- `cases_per_100k`
- `vaccination_pct`
- `case_fatality_rate`

A second socio-economic dataset is also used for unemployment, GDP growth, school closures, online learning, poverty rate, and internet access.

---

## ⚙️ Technologies Used

The project was developed using:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

## 📈 Features

This project includes:

- Loading COVID-19 and socio-economic datasets
- Data cleaning and preprocessing
- Handling missing values and duplicate rows
- Feature engineering
- Statistical summary of COVID-19 indicators
- Identification of pandemic waves using 7-day moving average
- Correlation analysis
- Data visualization with charts and heatmaps
- Linear Regression prediction model
- Interactive dashboard with Streamlit

---

## 📂 Project Structure

```text
Analiza-e-Perhapjes-se-COVID-19-dhe-Parashikimi-i-Rasteve/
│
├── data/
│   ├── clean/
│   │   ├── covid_kosova_CLEAN.csv
│   │   └── socioeconomic_kosova_CLEAN.csv
│   │
│   └── raw/
│       ├── covid_kosova_RAW.csv
│       └── socioeconomic_kosova_RAW.csv
│
├── outputs/
│   ├── 01_rastet_totale.png
│   ├── 02_rastet_ditore.png
│   ├── 03_vdekjet_totale.png
│   ├── 04_vaksinime.png
│   ├── 05_histogram.png
│   ├── 06_scatter_plot.png
│   ├── 07_papunesia.png
│   ├── 08_shkollimi.png
│   ├── 09_gdp_growth.png
│   ├── 10_heatmap.png
│   ├── 11_parashikimi_ml.png
│   └── 12_panel_permbledhes.png
│
├── src/
│   ├── analysis.py
│   ├── app.py
│   ├── load_data.py
│   ├── main.py
│   └── visualization.py
│
├── README.md
└── requirements.txt
```
##▶️ How to Run the Project

First, install the required Python libraries:
```
pip install -r requirements.txt
```
Then go to the src folder:
```
cd src
```
Run the main project:
```
python main.py
```
Or on Windows:
```
py main.py
```
To run the Streamlit dashboard:
```
streamlit run app.py
```
##📊 Outputs

After running main.py, the project generates 12 visualizations, including:

Total COVID-19 cases
Daily cases and 7-day moving average
Total deaths
Vaccination progress
Histogram of daily cases
Scatter plot of cases and deaths
Unemployment chart
Education impact chart
GDP growth chart
Correlation heatmap
Linear Regression prediction graph
Summary dashboard panel

👥 Team Members
Erdona Kadriolli
Yll Bytyqi
Fatlum Syla
