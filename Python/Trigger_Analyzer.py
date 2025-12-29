import pandas as pd
import numpy as np
import os

# ==========================================
# JOUW INSTELLINGEN
# ==========================================
mt4_files_path = "C:/MT4/MQL4/Files/"
tester_files_path = "C:/MT4/tester/files/"

PRICE_FILE = mt4_files_path + "TrainingData_Raw_EURUSD.csv"
# LET OP: PARAMS_FILE is weggehaald, want dat doet Market_Analyzer.py al!
CONFIG_FILE = tester_files_path + "JJ_Auto_Config.csv"

def calculate_pnl(row):
    # Target 1 = Buy Winst, 2 = Sell Winst
    if row['Target'] == 1: return 1.5 
    if row['Target'] == 2: return 1.5
    return -1.0 

def auto_train():
    print(">> 1. Data laden voor Trigger Optimalisatie...")
    if not os.path.exists(PRICE_FILE):
        print(f"ERROR: {PRICE_FILE} niet gevonden.")
        return

    df = pd.read_csv(PRICE_FILE, sep=';')
    
    # Filter metadata kolommen eruit om de indicatoren te vinden
    meta_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Target']
    all_indicators = [c for c in df.columns if c not in meta_cols]
    
    print(f">> {len(all_indicators)} indicatoren geladen.")
    
    # ID Map maken (nodig om straks de Trigger ID te vinden)
    id_map = {name: i for i, name in enumerate(all_indicators)}

    # ==========================================
    # 3. TRIGGER OPTIMALISATIE (BESTE M5)
    # ==========================================
    print(">> Beste Trigger zoeken uit ALLE M5 indicatoren...")
    
    df['PnL'] = df.apply(calculate_pnl, axis=1)
    
    # Pak alleen indicatoren op M5 timeframe
    m5_indicators = [c for c in all_indicators if "-M5" in c 
                     and "Session" not in c 
                     and "Day" not in c]
    
    best_trigger_name = "RSI-M5" # Default fallback
    best_trigger_pnl = -999.0
    best_min = 0.0
    best_max = 0.0
    
    for ind in m5_indicators:
        # We maken 10 bakjes (bins) van de indicator waarden
        try:
            df['Bin'] = pd.qcut(df[ind], 10, duplicates='drop')
        except:
            continue # Sla over als data te weinig variatie heeft
            
        stats = df.groupby('Bin', observed=False)['PnL'].mean()
        
        # Zoek het beste aaneengesloten bereik met positieve PnL
        current_pnl = stats.max()
        
        if current_pnl > best_trigger_pnl:
            best_bin = stats.idxmax()
            best_trigger_pnl = current_pnl
            best_trigger_name = ind
            best_min = best_bin.left
            best_max = best_bin.right
            
    print(f"   -> WINNAAR TRIGGER: {best_trigger_name} (Winstfactor: {best_trigger_pnl:.3f}R)")
    print(f"   -> Zone: {best_min:.4f} tot {best_max:.4f}")

    # ==========================================
    # 5. SEMI-AGNOSTISCHE FILTER OPTIMALISATIE
    # ==========================================
    print(">> Beste Volatiliteits-filter zoeken (BBW, StdDev, ATR)...")
    
    # De logische kandidaten (veiligheidskleppen)
    filter_types = ["BBW", "StdDev", "ATR"]
    
    # Zoek de kolomnamen in de CSV die hierbij horen (alleen M5)
    filter_candidates = []
    for f_type in filter_types:
        # Zoek kolom die 'f_type' bevat EN '-M5'
        found = [c for c in all_indicators if f_type in c and "-M5" in c]
        if found:
            filter_candidates.append(found[0]) # Pak de eerste match

    best_filter_name = "Geen"
    best_filter_max  = 999999.0
    best_filter_score = -999.0
    
    # We testen elke kandidaat
    for col in filter_candidates:
        # Stap 1: Bepaal de grens (90e percentiel van de winnende trades)
        wins = df[df['PnL'] > 0]
        if wins.empty: continue
            
        limit = wins[col].quantile(0.90)
        
        # Stap 2: Simuleer wat er gebeurt als we dit filter toepassen
        # We accepteren alleen trades ONDER de limiet
        filtered_df = df[df[col] < limit]
        
        # Bereken totaal resultaat na filteren
        total_pnl = filtered_df['PnL'].sum()
        
        # Is dit beter dan wat we hadden?
        if total_pnl > best_filter_score:
            best_filter_score = total_pnl
            best_filter_name  = col
            best_filter_max   = limit

    print(f"🛡️ WINNAAR FILTER: {best_filter_name}")
    print(f"   -> Grenswaarde: < {best_filter_max:.5f}")
    print(f"   -> Score na filter: {best_filter_score:.2f}R")

    # ID Ophalen voor de config
    filter_id = id_map.get(best_filter_name, -1)
    trigger_id = id_map.get(best_trigger_name, -1)

    # Wegschrijven Config
    with open(CONFIG_FILE, 'w') as f:
        f.write("Key;Value\n")
        f.write(f"Trigger_ID;{trigger_id}\n")
        f.write(f"Trigger_Name;{best_trigger_name}\n")
        f.write(f"Trigger_Min;{best_min}\n")
        f.write(f"Trigger_Max;{best_max}\n")
        # Nieuwe Filter Keys ipv BB_Max
        f.write(f"Filter_ID;{filter_id}\n")
        f.write(f"Filter_Name;{best_filter_name}\n")
        f.write(f"Filter_Max;{best_filter_max}\n")
        
    print(f">> Config opgeslagen in: {CONFIG_FILE}")

if __name__ == "__main__":
    auto_train()