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
    
    # Selecteer Top 5 per familie (Echt 15 stuks)
    top_osc   = final_params[final_params['Family'] == 0].sort_values('Weight', ascending=False).head(5)
    top_trend = final_params[final_params['Family'] == 1].sort_values('Weight', ascending=False).head(5)
    top_vol   = final_params[final_params['Family'] == 2].sort_values('Weight', ascending=False).head(5)


    diverse_top_15 = pd.concat([top_osc, top_trend, top_vol])

    # Filter alles wat NIET in de top 15 zit
    selected_names = diverse_top_15['Name'].tolist()
    rejected = final_params[~final_params['Name'].isin(selected_names)].sort_values('Weight', ascending=False)
    
    print("\n" + "="*50)
    print(f"--- AFGEWEZEN INDICATOREN (Totaal: {len(rejected)}) ---")
    print("="*50)
    
    # Forceer pandas om ALLE rijen te tonen (geen "..." afkapping)
    pd.set_option('display.max_rows', None)
    
    # Print de lijst netjes
    print(rejected[['Name', 'Family', 'Weight', 'Family']].to_string(index=False))
    
    # Reset de weergave-instelling voor het geval je hierna nog iets anders print
    pd.reset_option('display.max_rows')
    
    print("="*50 + "\n")
    
    # Normaliseer gewichten
    diverse_top_15['Weight'] = diverse_top_15['Weight'] / diverse_top_15['Weight'].sum()
    
# --- NIEUW: Bereken Threshold voor 5% trades ---
    print("Berekenen van thresholdwaarde...")
    
    # Initialiseer score array voor alle trainingsdata
    scores = np.zeros(len(train_df))
    
    # Loop alleen over de gekozen 15 indicatoren om de geaggregeerde score te bouwen
    for idx, row in diverse_top_15.iterrows():
        col_name = row['Name']
        if col_name in train_df.columns:
            # Haal waarden op
            values = train_df[col_name].values
            
            # Parameters
            w = row['Weight']
            d = row['Direction']
            m = row['Mean']
            s = row['StdDev']
            if s == 0: s = 1 # Voorkom deling door nul
            
            # Bereken weighted Z-score bijdrage
            z_score = (values - m) / s
            scores += (z_score * w * d)

    # Bepaal de threshold waarde (95e percentiel = top 5%)
    # Gebruik absolute waarden als trades beide kanten op kunnen, 
    # of percentiel 95 als het om directionele score gaat. 
    # Hier gaan we uit van de 'Long' trigger (95th percentile) als generieke threshold.
    calculated_threshold = np.percentile(scores, 95)
    
    print(f"Berekende Threshold (95%): {calculated_threshold:.5f}")
    
    # Voeg threshold toe aan dataframe (dezelfde waarde voor elke regel)
    diverse_top_15['Threshold'] = calculated_threshold
    
    # EXPORTEER
    output_columns = ['ID', 'Name', 'Weight', 'Mean', 'StdDev', 'Direction', 'Threshold']
    diverse_top_15[output_columns].to_csv(tester_files_path + "JJ_Scoring_Params.csv", sep=';', index=False)
    
    print(f"✅ SUCCES: Alleen de 15 top-indicatoren + Threshold zijn opgeslagen in {tester_files_path}JJ_Scoring_Params.csv")
    
    print(diverse_top_15[['ID', 'Name', 'Weight', 'Family']])

except Exception as e:
    print(f"Er ging iets mis: {e}")