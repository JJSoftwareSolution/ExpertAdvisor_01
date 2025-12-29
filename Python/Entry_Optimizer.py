import pandas as pd
import numpy as np
import traceback
import os

# --- INSTELLINGEN ---
mt4_files_path = "C:/MT4/MQL4/Files/"
tester_files_path = "C:/MT4/tester/files/"
symbol = "EURUSD"

# BELANGRIJK: Zet dit op True om H1 en H4 indicatoren te negeren
IGNORE_HTF = True

# Bestandsnamen
data_file   = mt4_files_path + f"TrainingData_Raw_{symbol}.csv"
params_file = tester_files_path + "JJ_Scoring_Params.csv"
stats_file  = mt4_files_path + f"Indicator_Stats_{symbol}.csv"
config_file = tester_files_path + "JJ_Trigger_Config.csv"

print(f"--- START INSTAP OPTIMALISATIE: {symbol} ---")
print("Doel: Zoek het beste filter uit MQL4 data om drawdown te verlagen.")

try:
    # ==============================================================================
    # 1. DATA & STATS LADEN
    # ==============================================================================
    print("1. Data laden...")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file niet gevonden: {data_file}")
        
    df = pd.read_csv(data_file, sep=';')
    df['Time'] = pd.to_datetime(df['Time'])

    # Stats laden (Bevat ALLE indicatoren uit MQL4)
    print(f"   Stats laden van: {stats_file}")
    stats_df = pd.read_csv(stats_file, sep=';')
    
    # Robuust indexeren: Name wordt de zoeksleutel
    if 'Name' in stats_df.columns:
        stats_df = stats_df.set_index('Name')
    
    # Params laden (Bevat de Top 15 van het Scoring Model)
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Params file niet gevonden: {params_file}. Draai eerst Market Analyzer.")
        
    params_df = pd.read_csv(params_file, sep=';').set_index('Name')

    # ==============================================================================
    # 2. SCORE RECONSTRUCTIE (Simulatie van de EA)
    # ==============================================================================
    print("2. EA Score simuleren...")
    df['AI_Score'] = 0.0

    for name, row in params_df.iterrows():
        if name not in df.columns: continue
        if name not in stats_df.index: continue

        mean = stats_df.loc[name, 'Mean']
        std  = stats_df.loc[name, 'StdDev']
        weight = row['Weight']
        direction = row['Direction']
        
        if std == 0: continue
        z_score = (df[name] - mean) / std
        z_score = z_score.clip(-3, 3)
        
        df['AI_Score'] += z_score * direction * weight
        
     # --- DE CORRECTIE: SCALING FACTOR TOEPASSEN ---
    # Net als in de EA vermenigvuldigen we de score met 5.
    # Hierdoor wordt 0.16 -> 0.80 en wordt de threshold van 0.7 gehaald.
    df['AI_Score'] *= 10.0 
    # ----------------------------------------------

    # ==============================================================================
    # 3. FILTER OP POTENTIËLE TRADES
    # ==============================================================================
    threshold = 0.7
    opportunities = df[df['AI_Score'] >= threshold].copy()

    if opportunities.empty:
        print("❌ Geen kansen gevonden met deze threshold. Is het model wel getraind?")
        exit()

    print(f"3. Analyse over {len(opportunities)} potentiële instapmomenten (Score > {threshold})")

    # ==============================================================================
    # 4. MAE BEREKENEN (De "Pijn" meting)
    # ==============================================================================
    horizon = 12
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
    opportunities['Future_Low'] = df['Low'].rolling(window=indexer).min()
    opportunities['MAE_Pips'] = (opportunities['Close'] - opportunities['Future_Low']) * 10000 

    # ==============================================================================
    # 5. TEST LIJST SAMENSTELLEN
    # ==============================================================================
    test_features = []
    
    all_indicators = list(stats_df.index)
    test_features += all_indicators
    
    test_features = list(set(test_features))
    test_features = [f for f in test_features if f in opportunities.columns]
    
    # FILTER HOGERE TIMEFRAMES ERUIT
    if IGNORE_HTF:
        original_count = len(test_features)
        test_features = [f for f in test_features if "H1" not in f and "H4" not in f]
        print(f"   -> Gefilterd: {original_count - len(test_features)} HTF indicatoren verwijderd.")
        
    print(f"   -> We testen {len(test_features)} indicatoren uit MQL4 op hun filter-kwaliteit.")

    # ==============================================================================
    # 6. CORRELATIE & IMPACT ANALYSE
    # ==============================================================================
    print("\n--- RESULTATEN: Welke indicator minimaliseert de pijn? ---")
    results = []

    for feat in test_features:
        if opportunities[feat].isnull().all(): continue
        if opportunities[feat].nunique() <= 1: continue

        median_val = opportunities[feat].median()
        
        mae_low_group  = opportunities[opportunities[feat] < median_val]['MAE_Pips'].mean()
        mae_high_group = opportunities[opportunities[feat] > median_val]['MAE_Pips'].mean()
        
        if pd.isna(mae_low_group) or pd.isna(mae_high_group): continue

        diff = abs(mae_high_group - mae_low_group)
        
        if mae_low_group < mae_high_group:
            ftype = "MAX" 
        else:
            ftype = "MIN"

        results.append({
            'Indicator': feat,
            'Impact_Pips': diff,
            'MAE_Low_Val': mae_low_group,
            'MAE_High_Val': mae_high_group,
            'Median': median_val,
            'Filter_Type': ftype
        })

    res_df = pd.DataFrame(results).sort_values('Impact_Pips', ascending=False)
    print(res_df[['Indicator', 'Impact_Pips', 'Filter_Type', 'Median']].head(10).to_string(index=False))

    # ==============================================================================
    # 7. EXPORT NAAR CONFIG (NU MET ID)
    # ==============================================================================
    if not res_df.empty:
        best_row = res_df.iloc[0] # De winnaar
        best_name = best_row['Indicator']

        # --- NIEUW: Zoek het ID op in de geladen stats ---
        best_id = -1
        if 'ID' in stats_df.columns:
            try:
                # Omdat 'Name' de index is, kunnen we direct .loc gebruiken
                best_id = int(stats_df.loc[best_name, 'ID'])
            except:
                best_id = -1
        # ------------------------------------------------

        print("\n--- ADVIES ---")
        print(f"De beste filter is: {best_name} (ID: {best_id})")
        
        best_mae = min(best_row['MAE_Low_Val'], best_row['MAE_High_Val'])
        worst_mae = max(best_row['MAE_Low_Val'], best_row['MAE_High_Val'])
        
        print(f"Instelling: {best_row['Filter_Type']} > {best_row['Median']:.6f}")
        print(f"Effect: Verlaagt gemiddelde drawdown van {worst_mae:.1f} naar {best_mae:.1f} pips.")
        
        # Wegschrijven in Key;Value formaat
        with open(config_file, 'w') as f:
            f.write("Key;Value\n")
            f.write(f"Filter_Name;{best_name}\n")
            f.write(f"Filter_ID;{best_id}\n")     # <--- TOEGEVOEGD
            f.write(f"Filter_Mode;{best_row['Filter_Type']}\n")
            f.write(f"Filter_Level;{best_row['Median']}\n")
            
        print(f"✅ Config opgeslagen in: {config_file}")
    else:
        print("❌ Geen geschikte resultaten gevonden om op te slaan.")

except Exception as e:
    print(f"❌ Fout opgetreden: {e}")
    traceback.print_exc()