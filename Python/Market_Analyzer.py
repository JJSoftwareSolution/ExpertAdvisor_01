import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
import argparse
import sys

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_osc", type=int, default=6)
    parser.add_argument("--n_trend", type=int, default=3)
    parser.add_argument("--n_vol", type=int, default=2)
    parser.add_argument("--red_threshold", type=float, default=0.5)
    parser.add_argument("--corr_h4", type=float, default=1.2)
    parser.add_argument("--corr_h1", type=float, default=1.0)
    parser.add_argument("--corr_m30", type=float, default=1.0)
    parser.add_argument("--corr_m5", type=float, default=1.0)
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
        
        # 2. MODEL & AGREEMENT (Kern-berekening)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_df.values, y)

        target_not_zero = (y != 0)
        total_signals = target_not_zero.sum()
        
        means_by_target = train_df.groupby('Target')[columns].mean()
        m_buy = means_by_target.loc[1].values if 1 in means_by_target.index else np.zeros(len(columns))
        m_sell = means_by_target.loc[2].values if 2 in means_by_target.index else np.zeros(len(columns))
        directions = np.where(m_buy > m_sell, 1, -1)
        
        m_all = X_df.values.mean(axis=0)
        s_all = X_df.values.std(axis=0)
        s_all[s_all == 0] = 1
        z_scores_full = (X_df.values - m_all) / s_all
        
        agreements = ((z_scores_full * directions > 0) & target_not_zero[:, np.newaxis]).sum(axis=0) / total_signals if total_signals > 0 else 0
        
        # 3. BASIS DATA SAMENSTELLEN
        results = pd.DataFrame({'Name': columns, 'Weight': rf.feature_importances_, 'Direction': directions, 'Agreement': agreements})
        
        # Alleen mergen met stats_df voor Family (nodig voor selectie)
        results = pd.merge(results, stats_df[['Name', 'Family', 'Mean', 'StdDev']], on='Name', how='inner')
        results['Eff_Score'] = results['Weight'] * results['Agreement']
        
        # 4. FILTERING & TIMEFRAME ADJ
        filtered = results[results['Agreement'] >= args.red_threshold].copy()
        filtered = filtered[~filtered['Name'].str.contains('Session|Day', case=False)]

        if args.use_tf_adj == "True":
            def adjust_eff(row):
                s, name = row['Eff_Score'], row['Name']
                if 'H4' in name: return s * args.corr_h4
                if 'M5' in name: return s * args.corr_m5
                return s
            filtered['Eff_Score'] = filtered.apply(adjust_eff, axis=1)

        # 5. SELECTIE
        diverse_top_15 = pd.concat([
            filtered[filtered['Family'] == 0].sort_values('Eff_Score', ascending=False).head(args.n_osc),
            filtered[filtered['Family'] == 1].sort_values('Eff_Score', ascending=False).head(args.n_trend),
            filtered[filtered['Family'] == 2].sort_values('Eff_Score', ascending=False).head(args.n_vol)
        ])
        diverse_top_15['Weight'] /= diverse_top_15['Weight'].sum()

        # 6. THRESHOLD LOOP (Dit is wat de Optimizer wil weten)
        top_names = diverse_top_15['Name'].values
        X_top = train_df[top_names].values
        z_scores_top = (X_top - diverse_top_15['Mean'].values) / np.where(diverse_top_15['StdDev'].values==0, 1, diverse_top_15['StdDev'].values)
        final_scores = (z_scores_top * diverse_top_15['Weight'].values * diverse_top_15['Direction'].values).sum(axis=1)
        conf_bits = (z_scores_top * diverse_top_15['Direction'].values > 0).astype(int)
        
        score_std = np.std(final_scores)
        best_overall_score, best_threshold = -1, 0
        final_idx = []

        for perc in np.arange(60.0, 99.8, 0.1):
            thresh = np.percentile(final_scores, perc)
            idx = np.where(final_scores >= thresh)[0]
            if len(idx) < 10: continue
            
            precision = (y[idx] == 1).sum() / len(idx)
            snr = (thresh / score_std)
            q_score = precision * snr * conf_bits[idx].mean()
            
            if q_score > best_overall_score:
                best_overall_score, best_threshold, final_idx = q_score, thresh, idx

        # 7. OUTPUT SECTIE
        if is_auto:
            # GEEN ballast, alleen de score voor de optimizer
            print(best_overall_score)
        else:
            # EXPORT CSV (alleen bij handmatige run)
            diverse_top_15['Threshold'] = best_threshold
            diverse_top_15['ID'] = range(len(diverse_top_15))
            diverse_top_15[['ID', 'Name', 'Weight', 'Mean', 'StdDev', 'Direction', 'Threshold']].to_csv(tester_files_path + "JJ_Scoring_Params.csv", sep=';', index=False)

            # UITGEBREIDE LIJST AFVALLERS (Zware berekening)
            log("\n❌ VOLLEDIGE LIJST AFGEVALLEN INDICATOREN:")
            top_base_names = set(diverse_top_15['Name'].str.split('-').str[0])
            dropped = results[~results['Name'].isin(diverse_top_15['Name'])].copy()
            
            def mark_dropped(row):
                name, base = row['Name'], row['Name'].split('-')[0]
                label = name
                if base not in top_base_names: label += "*"
                #if row['Agreement'] < args.red_threshold: label += " [LA]"
                return label

            dropped['Name'] = dropped.apply(mark_dropped, axis=1)
            
            pd.set_option('display.max_rows', None)
            print(dropped.sort_values('Weight', ascending=False)[['Name', 'Agreement', 'Weight', 'Eff_Score', 'Family']].to_string(index=False))
            pd.reset_option('display.max_rows')
                        
            log("\n" + "="*50)
            log(f"🎯 RESULTATEN (Threshold: {best_threshold:.5f})")
            log(f"Confluentie Score: {best_overall_score:.4f} | SNR: {best_threshold / score_std:.2f}x")
            total_days = (train_df['Time'].max() - train_df['Time'].min()).days
            log(f"Trades per jaar: {len(final_idx) / (total_days/365.25):.0f}")
            log("\n✅ GESELECTEERD:")
            print(diverse_top_15[['Name', 'Agreement', 'Weight', 'Eff_Score', 'Family']].to_string(index=False))

    except Exception as e:
        if is_auto: print("-1.0")
        else: print(f"❌ Fout: {e}")

if __name__ == "__main__":
    main()