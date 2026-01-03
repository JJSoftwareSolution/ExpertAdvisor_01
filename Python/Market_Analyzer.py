import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
import argparse
import sys

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    # M30 Swing settings
    parser.add_argument("--n_trend_high", type=int, default=3) 
    parser.add_argument("--n_osc_mid", type=int, default=5)   
    parser.add_argument("--n_osc_fast", type=int, default=3)   # M5 Timing
    parser.add_argument("--n_vol", type=int, default=2)        # Family 4 selector
    
    parser.add_argument("--red_threshold", type=float, default=0.5)
    parser.add_argument("--corr_h4", type=float, default=0.8)
    parser.add_argument("--corr_h1", type=float, default=1.0)
    parser.add_argument("--corr_m30", type=float, default=1.0)
    parser.add_argument("--corr_m5", type=float, default=1.5)
    parser.add_argument("--use_tf_adj", type=str, default="True")
    parser.add_argument("--use_fam_sel", type=str, default="True")
    args = parser.parse_args()

    is_auto = len(sys.argv) > 1
    mt4_files_path = "C:/MT4/MQL4/Files/"
    tester_files_path = "C:/MT4/tester/files/"
    symbol = "EURUSD"
    split_date = '2024-01-01'
    
    def log(msg):
        if not is_auto: print(msg)

    try:
        # 1. DATA INLADEN
        stats_df = pd.read_csv(mt4_files_path + f"Indicator_Stats_{symbol}.csv", sep=';')
        df = pd.read_csv(mt4_files_path + f"TrainingData_Raw_{symbol}.csv", sep=';').dropna(axis=1, how='all')
        df['Time'] = pd.to_datetime(df['Time'])
        
        train_df = df[df['Time'] < split_date].copy()
        y = train_df['Target'].values
        X_df = train_df.drop(['Time', 'Target'], axis=1)
        columns = X_df.columns
        
        # 2. MODEL & AGREEMENT
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_df.values, y)

        target_not_zero = (y != 0)
        total_signals = target_not_zero.sum()
        
        # --- FIX: Betere richtingbepaling via Correlatie ---
        # We maken een tijdelijke target kolom: 1 = Buy, -1 = Sell, 0 = Niks
        # Dit helpt om te zien of een indicator écht correleert met Buy of Sell
        temp_target = train_df['Target'].copy()
        temp_target = temp_target.replace(2, -1) 
        
        directions = []
        for col in columns:
            # Bereken correlatie tussen indicator en richting (-1 of 1)
            corr = train_df[col].corr(temp_target)
            # Als correlatie positief is, is Direction 1. Anders -1.
            # Dit lost het '28% precision' probleem op.
            directions.append(1 if corr >= 0 else -1)
        directions = np.array(directions)
        
        m_all = X_df.values.mean(axis=0)
        s_all = X_df.values.std(axis=0)
        s_all[s_all == 0] = 1
        z_scores_full = (X_df.values - m_all) / s_all
        
        agreements = ((z_scores_full * directions > 0) & target_not_zero[:, np.newaxis]).sum(axis=0) / total_signals if total_signals > 0 else 0
        
        # 3. BASIS DATA SAMENSTELLEN
        results = pd.DataFrame({'Name': columns, 'Weight': rf.feature_importances_, 'Direction': directions, 'Agreement': agreements})
        results = pd.merge(results, stats_df[['Name', 'Family', 'Mean', 'StdDev']], on='Name', how='inner')
        
        # --- [START WIJZIGING] LOGICA VOOR FAMILY 4 (NON-DIRECTIONAL) ---
        fam4_mask = results['Family'] == 4
        if fam4_mask.any():
            log(f">> Verwerken van {fam4_mask.sum()} Type 4 indicatoren (Lift Analysis)...")
            baseline_prob = target_not_zero.mean() # Hoe vaak beweegt de markt gemiddeld?
            
            for idx, row in results[fam4_mask].iterrows():
                col_name = row['Name']
                if col_name not in X_df.columns: continue
                
                # Bepaal wanneer indicator actief is (> gemiddelde)
                col_mean = X_df[col_name].mean()
                is_active = X_df[col_name] > col_mean
                
                # Lift: Kans op beweging als indicator actief is
                if is_active.sum() > 10: # Minimaal aantal samples
                    prob_given_active = target_not_zero[is_active].mean()
                    
                    if baseline_prob > 0:
                        lift = (prob_given_active - baseline_prob) / baseline_prob
                    else:
                        lift = 0
                    
                    # Overschrijf de RF Weight met de Lift score (factor 5.0 voor schaling)
                    new_weight = max(0, lift * 5.0)
                    results.at[idx, 'Weight'] = new_weight
                    
                    # Forceer Agreement op 1.0 (zodat hij niet wordt weggefilterd in stap 4)
                    # We gebruiken Lift als bewijs van nut, niet richtings-overeenkomst
                    results.at[idx, 'Agreement'] = 1.0
                    
                    # Direction zetten we op 1 voor export (EA gebruikt dubbele tel-logica)
                    results.at[idx, 'Direction'] = 1 
        # --- [EINDE WIJZIGING] ---

        results['Eff_Score'] = results['Weight'] * results['Agreement']
        
        # 4. FILTERING
        filtered = results[results['Agreement'] >= args.red_threshold].copy()
        filtered = filtered[~filtered['Name'].str.contains('Session|Day', case=False)] # Tijd-filters apart

        if args.use_tf_adj == "True":
            def adjust_eff(row):
                s, name = row['Eff_Score'], row['Name']
                if 'H4' in name: return s * args.corr_h4
                if 'M5' in name: return s * args.corr_m5
                return s
            filtered['Eff_Score'] = filtered.apply(adjust_eff, axis=1)

        # 5. SELECTIE
        
        # A. TREND (H4/H1)
        trend_candidates = filtered[
            (filtered['Family'] == 1) & (filtered['Name'].str.contains('-H4|-H1'))
        ].sort_values('Eff_Score', ascending=False).head(args.n_trend_high).copy()
        
        # B. SWING (M30)
        osc_mid_candidates = filtered[
            (filtered['Family'] == 0) & (filtered['Name'].str.contains('-M30'))
        ].sort_values('Eff_Score', ascending=False).head(args.n_osc_mid).copy()

        # C. TIMING (M5)
        osc_fast_candidates = filtered[
            (filtered['Family'] == 0) & (filtered['Name'].str.contains('-M5'))
        ].sort_values('Eff_Score', ascending=False).head(args.n_osc_fast).copy()

        # D. VOLATILITY / CONTEXT (Family 4)
        vol_candidates = pd.DataFrame()
        if args.n_vol > 0:
            vol_candidates = filtered[
                (filtered['Family'] == 4)
            ].sort_values('Eff_Score', ascending=False).head(args.n_vol).copy()

        # Normaliseer en pas nieuwe verdeling toe
        if not trend_candidates.empty:
            trend_candidates['Weight'] = (trend_candidates['Weight'] / trend_candidates['Weight'].sum()) * 0.35
            
        if not osc_mid_candidates.empty:
            osc_mid_candidates['Weight'] = (osc_mid_candidates['Weight'] / osc_mid_candidates['Weight'].sum()) * 0.35
            
        if not osc_fast_candidates.empty:
            osc_fast_candidates['Weight'] = (osc_fast_candidates['Weight'] / osc_fast_candidates['Weight'].sum()) * 0.20
            
        if not vol_candidates.empty:
            vol_candidates['Weight'] = (vol_candidates['Weight'] / vol_candidates['Weight'].sum()) * 0.10

        # Samenvoegen (Nu met 4 groepen)
        diverse_top_15 = pd.concat([trend_candidates, osc_mid_candidates, osc_fast_candidates, vol_candidates])
        
        # Check: Telt het op tot 1.0?
        diverse_top_15['Weight'] /= diverse_top_15['Weight'].sum()

        # 6. THRESHOLD LOOP (CRASH FIX & SAFETY)
        top_names = diverse_top_15['Name'].values
        X_top = train_df[top_names].values
        
        # Z-Scores berekenen
        z_scores_top = (X_top - diverse_top_15['Mean'].values) / np.where(diverse_top_15['StdDev'].values==0, 1, diverse_top_15['StdDev'].values)
        
        # Totale score per bar
        final_scores = (z_scores_top * diverse_top_15['Weight'].values * diverse_top_15['Direction'].values).sum(axis=1)
        
        score_std = np.std(final_scores)
        if score_std == 0: score_std = 1 
        
        total_days = (train_df['Time'].max() - train_df['Time'].min()).days
        if total_days <= 0: total_days = 1
        
        # DEFAULT INSTELLINGEN (Voorkomt crash als alles faalt)
        best_overall_score = -1.0
        best_threshold = 0.5 
        final_idx = []     # <--- FIX: Lege lijst als startpunt
        
        max_precision_found = 0.0

        # Scan loop
        for perc in np.arange(60.0, 99.9, 0.2):
            thresh = np.percentile(final_scores, perc)
            
            # Negeer thresholds rond 0 (ruis)
            if thresh < 0.05: continue
            
            idx = np.where(final_scores >= thresh)[0]
            
            # Minimaal 20 trades per jaar
            current_trades_per_year = len(idx) / (total_days / 365.25)
            if len(idx) < 20 or current_trades_per_year < 20: continue 
            
            precision = (y[idx] == 1).sum() / len(idx)
            if precision > max_precision_found: max_precision_found = precision
            
            # EIS: Moet beter zijn dan gokken (50.1%)
            if precision <= 0.501: continue 
            
            edge = precision - 0.50
            
            # Score Formule
            raw_score = (edge ** 2) * np.sqrt(current_trades_per_year) * 1000
            
            if raw_score > best_overall_score:
                best_overall_score, best_threshold, final_idx = raw_score, thresh, idx

        # FALLBACK: Als final_idx nog steeds leeg is (geen goede threshold gevonden)
        if len(final_idx) == 0:
             # We vullen hem op basis van de default threshold, puur om errors te voorkomen
             final_idx = np.where(final_scores >= best_threshold)[0]
             
             if is_auto:
                 pass
             elif best_overall_score == -1.0:
                 print(f"⚠️ WAARSCHUWING: Geen winstgevende threshold gevonden! (Max precision: {max_precision_found:.2%})")
                 print(f"   -> Systeem gebruikt default threshold: {best_threshold}")
                 
                 
        # 7. OUTPUT SECTIE
        if is_auto:
            print(best_overall_score)
        else:
            diverse_top_15['Threshold'] = best_threshold
            diverse_top_15['ID'] = range(len(diverse_top_15))
            diverse_top_15[['ID', 'Name', 'Weight', 'Mean', 'StdDev', 'Direction', 'Threshold']].to_csv(tester_files_path + "JJ_Scoring_Params.csv", sep=';', index=False)

            log("\n❌ VOLLEDIGE LIJST AFGEVALLEN INDICATOREN:")
            top_base_names = set(diverse_top_15['Name'].str.split('-').str[0])
            dropped = results[~results['Name'].isin(diverse_top_15['Name'])].copy()
            
            def mark_dropped(row):
                name = row['Name']
                base = name.split('-')[0]
                label = name
                if base not in top_base_names: label += "*"
                if 'Agreement' in row and row['Agreement'] < args.red_threshold: label += " [LA]"
                return label

            dropped['Name'] = dropped.apply(mark_dropped, axis=1)
            
            pd.set_option('display.max_rows', None)
            cols_to_print = ['Name', 'Agreement', 'Weight', 'Eff_Score', 'Family']
            print(dropped.sort_values('Weight', ascending=False)[cols_to_print].to_string(index=False))
            pd.reset_option('display.max_rows')
                        
            log("\n" + "="*50)
            log(f"🎯 RESULTATEN (Threshold: {best_threshold:.5f})")
            log(f"Winst Score: {best_overall_score:.4f}")
            log(f"SNR: {best_threshold / score_std:.2f}x")
            log(f"Trades per jaar: {len(final_idx) / (total_days/365.25):.0f}")
            log("\n✅ GESELECTEERD (INC. FAMILY 4):")
            print(diverse_top_15[['Name', 'Agreement', 'Weight', 'Eff_Score', 'Family']].to_string(index=False))

    except Exception as e:
        if is_auto: print("-1.0")
        else: print(f"❌ Fout: {e}")

if __name__ == "__main__":
    main()