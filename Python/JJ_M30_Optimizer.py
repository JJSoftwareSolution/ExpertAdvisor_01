import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
import sys

# Onderdruk warnings voor schone output
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATIE & CONSTANTEN
# ==============================================================================
class Config:
    # Paden (pas aan indien nodig)
    PATH_DATA = "C:/MT4/MQL4/Files/TrainingData_Raw_EURUSD.csv"
    PATH_STATS = "C:/MT4/MQL4/Files/Indicator_Stats_EURUSD.csv"
    
    # SETUP voor DEMO-account
#    PATH_OUTPUT = "C:/MT4/tester/files/JJ_Daily_Plan.csv"
#    TRAIN_END_DATE = '2022-12-31'
#    TEST_START_DATE = '2023-01-01'
    # SETUP voor LIVE-account
    PATH_OUTPUT = "C:/MT4/MQL4/files/JJ_Daily_Plan.csv"
    TRAIN_END_DATE = '2025-12-31'
    TEST_START_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Model Settings
    MIN_CONFIDENCE = 0.65       # Alleen trades met >65% model confidence
    MIN_VOTES = 2               # Minimaal 2 indicatoren in consensus (Cluster Logic)
    
    # Risk Management
    MAX_TRADES_DAILY = 2        # Max 2 trades per dag
    MIN_HOURS_BETWEEN = 4       # Minimaal 4 uur tussen trades
    MIN_ADX = 20.0              # Family 4 Filter: Geen trades in dode markt
    
    # Target Mapping
    TARGET_BUY = 1
    TARGET_SELL = 2

# ==============================================================================
# KLASSE: SYSTEM CORE
# ==============================================================================
class EuroDollarSystem:
    def __init__(self):
        self.df = None
        self.stats = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_meta = {} # Opslag voor richting en familie per kolom
        
    def load_data(self):
        print(">> 1. Data Laden & Structureren...")
        try:
            # Laad metadata
            self.stats = pd.read_csv(Config.PATH_STATS, sep=';')
            # Maak map van Name -> Family
            self.fam_map = dict(zip(self.stats['Name'], self.stats['Family']))
            
            # Laad trainingsdata
            self.df = pd.read_csv(Config.PATH_DATA, sep=';').dropna(axis=1, how='all')
            self.df['Time'] = pd.to_datetime(self.df['Time'])
            
            # Filter kolommen die we niet hebben in metadata (veiligheid)
            valid_cols = [c for c in self.df.columns if c in self.fam_map or c in ['Time', 'Target']]
            self.df = self.df[valid_cols]
            
            print(f"   - Rows: {len(self.df)}")
            print(f"   - Features: {len(valid_cols)-2}")
            
        except FileNotFoundError as e:
            print(f"❌ FOUT: Bestand niet gevonden. {e}")
            sys.exit(1)

    def analyze_market_structure(self):
        print(">> 2. Markt Structuur Analyse (Correctie Richting - STRICT)...")
        
        # STAP 1: Filter EERST op trainingsdata!
        # We mogen alleen kijken naar data tot en met 2023
        train_mask = self.df['Time'] <= Config.TRAIN_END_DATE
        train_df = self.df[train_mask]
        
        # Maak tijdelijke target alleen voor de trainingsset
        temp_target = train_df['Target'].copy()
        temp_target = temp_target.replace(Config.TARGET_SELL, -1)
        
        feature_cols = [c for c in self.df.columns if c not in ['Time', 'Target']]
        
        print(f"   - Analyse gebaseerd op {len(train_df)} rijen t/m {Config.TRAIN_END_DATE}")
        
        for col in feature_cols:
            # Correlatie berekenen op ALLEEN trainingsdata
            corr = train_df[col].corr(temp_target)
            
            # Als er geen correlatie is (bv constante 0), zet op 0
            if pd.isna(corr): corr = 0
            
            direction = 1 if corr >= 0 else -1
            family = self.fam_map.get(col, -1)
            
            self.feature_meta[col] = {
                'direction': direction,
                'family': family,
                'weight': abs(corr) * 10 
            }

    def train_ensemble_model(self):
        print(">> 3. Training Ensemble Model...")
        
        # Split Data
        train_mask = self.df['Time'] <= Config.TRAIN_END_DATE
        train_df = self.df[train_mask]
        
        X = train_df.drop(['Time', 'Target'], axis=1)
        y = train_df['Target']
        
        self.feature_names = X.columns.tolist()
        
        # We trainen alleen op rijen waar Target != 0 (Actie)
        active_mask = y != 0
        X_train = X[active_mask]
        y_train = y[active_mask]
        
        # Schaal de data
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        # ENSEMBLE: Random Forest + Gradient Boosting
        # Dit zorgt voor minder overfitting dan 1 enkel model
        rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        
        self.model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
        self.model.fit(X_scaled, y_train)
        
        print(f"   - Model getraind. Klaar voor validatie vanaf {Config.TEST_START_DATE}.")



    # ==============================================================================
    # NIEUWE FUNCTIE: TOON NUTTELOZE INDICATOREN
    # ==============================================================================
    def analyze_feature_importance(self):
        print("\n>> 3a. Feature Importance Analyse...")
        
        # Haal de getrainde sub-modellen op uit de VotingClassifier
        # index 0 = RandomForest, index 1 = GradientBoosting
        rf_model = self.model.estimators_[0]
        gb_model = self.model.estimators_[1]
        
        # Haal de scores op
        rf_imp = rf_model.feature_importances_
        gb_imp = gb_model.feature_importances_
        
        # Bereken het gemiddelde belang
        avg_imp = (rf_imp + gb_imp) / 2
        
        # Maak een dataframe voor overzicht
        imp_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': avg_imp
        }).sort_values(by='Importance', ascending=False)
        
        # Filter 1: Indicatoren met EXACT 0.0 invloed (kunnen direct weg)
        zero_impact = imp_df[imp_df['Importance'] == 0.0]
        
        # Filter 2: Indicatoren met verwaarloosbare invloed (< 0.1%)
        low_impact = imp_df[(imp_df['Importance'] > 0.0) & (imp_df['Importance'] < 0.001)]
        
        print(f"   - Totaal aantal features: {len(self.feature_names)}")
        print(f"   - 🗑️ Features met 0.0% invloed (KAN JE VERWIJDEREN): {len(zero_impact)}")
        
        if not zero_impact.empty:
            print("\n   ⚠️  DEZE INDICATOREN HEBBEN GEEN ENKEL EFFECT:")
            print(zero_impact['Feature'].to_string(index=False))
            
        print(f"\n   - 📉 Features met zeer lage invloed (< 0.1%): {len(low_impact)}")
        if not low_impact.empty:
            print("   (Overweeg deze ook te verwijderen voor een schoner model)")
            print(low_impact[['Feature', 'Importance']].head(10).to_string(index=False))
            
        print("\n   - 🏆 Top 5 Belangrijkste features:")
        print(imp_df.head(5).to_string(index=False))
        print("-" * 50 + "\n")


    def run_daily_workflow(self):
        print(f">> 4. Uitvoeren Dagelijkse Workflow (High Performance Mode)...")

        # Als TEST_START_DATE in het verleden ligt (vóór vandaag), gebruiken we die datum.
        # Als het vandaag is, gebruiken we de 'laatste 7 dagen' logica voor het weekend of als beurs meerdere dagen was gesloten.
        
        target_start = pd.to_datetime(Config.TEST_START_DATE)
        today = pd.Timestamp(datetime.now().date())

        if target_start < today:
            # Mode: Backtest / Testset maken
            start_time = target_start
        else:
            # Mode: Live Trading (weekend proof)
            last_date_in_data = self.df['Time'].max()
            start_time = last_date_in_data - timedelta(days=7)
        
        # Test Data Selecteren
        test_mask = self.df['Time'] >= start_time
        test_df = self.df[test_mask].copy().reset_index(drop=True)
        
        # --- VEILIGHEIDSCHECK VOOR LEGE DATA ---
        if test_df.empty:
            print(f"\n❌ KRITISCHE FOUT: Geen data gevonden vanaf {Config.TEST_START_DATE}!")
            print(f"   De laatste datum in je CSV is: {self.df['Time'].max()}")
            print("   👉 OPLOSSING: Draai je MT4 'JJ_DataCollector' script opnieuw met een datum in de toekomst (bv 2026) en COMPILEER het script eerst!\n")
            sys.exit(1)
        # ---------------------------------------
        
        # Features schalen (in één keer)
        X_test = test_df.drop(['Time', 'Target'], axis=1)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("   - A. Model voorspellingen berekenen...")
        probs = self.model.predict_proba(X_test_scaled)
        
        # ---------------------------------------------------------
        # STAP A: VECTORISATIE (Reken alles vooraf uit)
        # ---------------------------------------------------------
        print("   - B. Indicatoren consensus berekenen (Vectorized)...")
        
        # 1. Signalen uit Model
        # We maken kolommen met True/False voor Buy en Sell signalen
        buy_signal_mask = probs[:, 0] > Config.MIN_CONFIDENCE
        sell_signal_mask = probs[:, 1] > Config.MIN_CONFIDENCE
        
        # 2. ADX Filter (Family 4)
        # Zoek de ADX kolom
        adx_col = [c for c in X_test.columns if 'ADX' in c and 'M30' in c][0]
        adx_ok_mask = X_test[adx_col] >= Config.MIN_ADX
        
        # 3. Consensus Scores Berekenen (Zonder loop over rijen!)
        # We bouwen een 'Buy_Score' en 'Sell_Score' voor elke rij
        buy_votes = np.zeros(len(test_df))
        sell_votes = np.zeros(len(test_df))
        
        # Haal kolommen op
        m30_cols = [c for c in X_test.columns if 'M30' in c and self.feature_meta[c]['family'] in [0, 1]]
        h4_cols = [c for c in X_test.columns if 'H4' in c and self.feature_meta[c]['family'] == 1]
        
        # Loop door de KOLOMMEN (100x) ipv RIJEN (100.000x) -> Dit is 1000x sneller
        for col in m30_cols + h4_cols:
            vals = X_test[col].values
            direction = self.feature_meta[col]['direction']
            
            # Als direction 1 is: Positieve waarde = BUY
            # Als direction -1 is: Negatieve waarde = BUY
            if direction == 1:
                is_buy = vals > 0
                is_sell = vals < 0
            else:
                is_buy = vals < 0
                is_sell = vals > 0
            
            buy_votes += is_buy.astype(int)
            sell_votes += is_sell.astype(int)

        # 4. H4 Filter (Specifieke check)
        h4_buy_ok = np.zeros(len(test_df), dtype=bool)
        h4_sell_ok = np.zeros(len(test_df), dtype=bool)
        
        for col in h4_cols:
            vals = X_test[col].values
            direction = self.feature_meta[col]['direction']
            # Logica: Als signaal BUY is, moet H4 trend BUY zijn (of neutraal, maar niet SELL)
            # Hier eisen we dat er minimaal 1 H4 indicator mee eens is
            if direction == 1:
                h4_buy_ok |= (vals > 0)
                h4_sell_ok |= (vals < 0)
            else:
                h4_buy_ok |= (vals < 0)
                h4_sell_ok |= (vals > 0)

        # ---------------------------------------------------------
        # STAP B: KANDIDATEN SELECTIE
        # ---------------------------------------------------------
        print("   - C. Kandidaten filteren...")
        
        # Combineer alle voorwaarden in één masker
        # Kandidaat BUY = (Model Buy) EN (ADX OK) EN (Genoeg Votes) EN (H4 Trend OK)
        final_buy_candidates = (
            buy_signal_mask & 
            adx_ok_mask & 
            (buy_votes >= Config.MIN_VOTES) & 
            h4_buy_ok
        )
        
        final_sell_candidates = (
            sell_signal_mask & 
            adx_ok_mask & 
            (sell_votes >= Config.MIN_VOTES) & 
            h4_sell_ok
        )
        
        # Voeg samen en pak de indexen
        candidates_idx = np.where(final_buy_candidates | final_sell_candidates)[0]
        
        print(f"   - {len(candidates_idx)} potentiële setups gevonden. Nu tijd-filters toepassen...")

        # ---------------------------------------------------------
        # STAP C: TIJD MANAGEMENT (De enige loop die overblijft)
        # ---------------------------------------------------------
        results = []
        last_trade_time = pd.Timestamp("2000-01-01") # Veilige startdatum
        trades_today = 0
        current_day = None
        
        # We loopen nu alleen over de kandidaten (bv. 500 rijen ipv 100.000)
        for i in candidates_idx:
            timestamp = test_df.at[i, 'Time']
            
            # Dag reset
            if current_day != timestamp.date():
                current_day = timestamp.date()
                trades_today = 0
                
            # Max trades check
            if trades_today >= Config.MAX_TRADES_DAILY: continue
            
            # Rusttijd check
            if (timestamp - last_trade_time) < timedelta(hours=Config.MIN_HOURS_BETWEEN): continue
            
            # --- TRADE ACCEPTED ---
            # Bepaal richting
            if final_buy_candidates[i]:
                entry_type = "BUY"
                confidence = probs[i][0]
            else:
                entry_type = "SELL"
                confidence = probs[i][1]
            
            # ATR ophalen
            atr_col = [c for c in X_test.columns if 'ATR-M30' in c][0]
            atr_val = test_df.at[i, atr_col]
            
            sl_pips = atr_val * 1.0
            tp_pips = atr_val * 1.5
            
            # Reden opbouwen (Cluster info) - Dit doen we alleen voor de final trades
            row = test_df.iloc[i]
            reasons = []
            
            # Snel de redenen ophalen voor display
            check_cols = m30_cols
            for c in check_cols:
                val = row[c]
                direction = self.feature_meta[c]['direction']
                is_agree = False
                if entry_type == "BUY":
                    is_agree = (val > 0) if direction == 1 else (val < 0)
                else:
                    is_agree = (val < 0) if direction == 1 else (val > 0)
                
                if is_agree: reasons.append(c.split('-')[0])

            reason_str = "+".join(list(set(reasons))[:3])
            
            results.append({
                'Time': timestamp,
                'Direction': entry_type,
                'Confidence': round(confidence, 2),
                'Cluster': reason_str,
                'ATR_M30': round(atr_val, 5),
                'SL_Dist': round(sl_pips, 5),
                'TP_Dist': round(tp_pips, 5),
                'M5_Refine': "Check M5"
            })
            
            # Update state
            last_trade_time = timestamp
            trades_today += 1
            
        return pd.DataFrame(results)

    def save_results(self, df_results):
        if df_results.empty:
            print("⚠️ Geen trades gevonden die aan de strenge eisen voldoen.")
        else:
            print(f">> 5. Resultaten Opslaan: {len(df_results)} setups gevonden.")
            print(df_results.head(10).to_string(index=False))
            df_results.to_csv(Config.PATH_OUTPUT, sep=';', index=False)
            print(f"✅ Opgeslagen in: {Config.PATH_OUTPUT}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- CHECK TRAIN_END_DATE ---
    train_date = datetime.strptime(Config.TRAIN_END_DATE, '%Y-%m-%d')
    drie_maanden_geleden = datetime.now() - timedelta(days=90)

    if train_date < drie_maanden_geleden:
        print(f"\n!!! LET OP: De trainingsdata (t/m {Config.TRAIN_END_DATE}) is meer dan 3 maanden oud.")
        print("   Overweeg om de TRAIN_END_DATE bij te werken voor een recenter model.\n")
    # ----------------------------
    
    system = EuroDollarSystem()
    
    # 1. Load
    system.load_data()
    
    # 2. Analyze (Correct Direction)
    system.analyze_market_structure()
    
    # 3. Train
    system.train_ensemble_model()
    
    # --- NIEUW: Bekijk welke indicatoren weg kunnen ---
    system.analyze_feature_importance()
    # --------------------------------------------------
    
    # 4. Predict & Filter
    trade_plan = system.run_daily_workflow()
    
    # 5. Export
    system.save_results(trade_plan)