# STATCAL ONLINE IDX Banking Financial Descriptive and Correlation Analyzer

This Streamlit application analyzes IDX banking financial data with a lighter feature set.

## Main features

- Upload Excel file or use included sample data
- Filter data by Year and Ticker Code
- Univariate descriptive statistics
- Grouped descriptive statistics
- Correlation matrix with p-values
- Publication-ready correlation heatmap
- Export correlation heatmap to PNG
- Export tables to Excel
- Includes st.cache_data for data loading and statistical computation

## Training Data

https://drive.google.com/drive/folders/1s273Ad5FUElhzd5G16jWSBxbOtforzRR?usp=sharing

## Run

```bash
pip install -r requirements.txt
python -m streamlit run main.py
```
