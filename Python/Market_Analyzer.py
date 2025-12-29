import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. INSTELLINGEN ---
mt4_files_path = "C:/MT4/MQL4/Files/"
tester_files_path = "C:/MT4/tester/files/"

data_file  = mt4_files_path + "TrainingData_Raw_EURUSD.csv"
stats_file = mt4_files_path + "Indicator_Stats_EURUSD.csv"

print("--- START ANALYSE ---")

try:
    stats_df = pd.read_csv(stats_file, sep=';')
    df = pd.read_csv(data_file, sep=';')
    df = df.dropna(axis=1, how='all')
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Split op 2024
    split_date = '2024-01-01'
    train_df = df[df['Time'] < split_date]
    
    print(f"Trainingsregels: {len(train_df)}")

    y = train_df['Target']
    X = train_df.drop(['Time', 'Target'], axis=1)
    
    # Model training
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    results = pd.DataFrame({'Name': X.columns, 'Weight': rf.feature_importances_})

    # Richting bepalen
    directions = []
    for col in X.columns:
        mean_buy  = train_df[train_df['Target'] == 1][col].mean()
        mean_sell = train_df[train_df['Target'] == 2][col].mean()
        directions.append(1 if mean_buy > mean_sell else -1)
    results['Direction'] = directions

    # Samenvoegen met MT4 stats
    final_params = pd.merge(results, stats_df, on='Name')

    # Filter Session/Day
    ignore_list = ['Session', 'Day'] 
    mask = final_params['Name'].str.contains('|'.join(ignore_list), case=False)
    final_params = final_params[~mask]

    # Timeframe Penalty/Boost
    def adjust_weight(row):
        w = row['Weight']
        if 'H4' in row['Name']: return w * 0.5
        if 'H1' in row['Name']: return w * 0.7
        if 'M5' in row['Name']: return w * 1.5
        return w

    final_params['Weight'] = final_params.apply(adjust_weight, axis=1)

    # Familie bepalen
    def get_family_by_type(name):
        if any(x in name for x in ['Trend_200', 'MACD', 'ADX', 'SAR', 'Mom', 'MADist']): return 1
        if any(x in name for x in ['RSI', 'Stoch', 'CCI', 'WPR', 'MFI', 'OsMA', 'DeM', 'RVI']): return 0
        # TOEVOEGING: 'BB' en 'Width' zodat Bollinger Bands meedoen
        if any(x in name for x in ['OBV', 'AD', 'Force', 'BullP', 'BearP', 'StdDev', 'BBW']): return 2
        return 3

    
    final_params['Family'] = final_params['Name'].apply(get_family_by_type)
    
    # Selecteer Top 5 per familie (Echt 15 stuks)
    top_osc   = final_params[final_params['Family'] == 0].sort_values('Weight', ascending=False).head(5)
    top_trend = final_params[final_params['Family'] == 1].sort_values('Weight', ascending=False).head(5)
    top_vol   = final_params[final_params['Family'] == 2].sort_values('Weight', ascending=False).head(5)
    
    diverse_top_15 = pd.concat([top_osc, top_trend, top_vol])
    diverse_top_15['Weight'] = diverse_top_15['Weight'] / diverse_top_15['Weight'].sum()
    
    # EXPORTEER ALLEEN DEZE 15 (Verwijder alle andere export regels onderaan!)
    output_columns = ['ID', 'Weight', 'Mean', 'StdDev', 'Direction', 'Name']
    diverse_top_15[output_columns].to_csv(tester_files_path + "JJ_Scoring_Params.csv", sep=';', index=False)
    
    print(f"✅ SUCCES: Alleen de 15 top-indicatoren zijn opgeslagen in {tester_files_path}JJ_Scoring_Params.csv")
    
    print(diverse_top_15[['ID', 'Name', 'Weight']])

except Exception as e:
    print(f"Er ging iets mis: {e}")