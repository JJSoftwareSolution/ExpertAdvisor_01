import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Advies van Gemini
# 
# De beste resultaten komen meestal uit deze configuratie:
# Instelling                  Waarde  Waarom?
# USE_TIMEFRAME_ADJUSTMENT    False   Voorkomt kunstmatige bias naar ruisige M5 data.
# USE_FAMILY_SELECTION        True    Dwingt diversiteit af; robuuster in wisselende markten.
# Aantal indicatoren          15      Optimale balans tussen ruisfiltering en signaalsterkte.
# Timeframe correctie                 N.v.t.Laat de RF-algoritme het zware werk doen.

# --- 1. INSTELLINGEN ---
mt4_files_path = "C:/MT4/MQL4/Files/"
tester_files_path = "C:/MT4/tester/files/"
symbol = "EURUSD"
DESIRED_SIGNALS_PER_DAY = 10  # Richtlijn voor het aantal signalen per dag
split_date = '2024-01-01'

USE_REDUNDANTIE_FILTER   = True
USE_TIMEFRAME_ADJUSTMENT = True  # True = penalties/boosts, False = pure weights
USE_FAMILY_SELECTION     = True      # True = verdeeld over families, False = puur de beste 15 totaal

data_file  = mt4_files_path + f"TrainingData_Raw_{symbol}.csv"
stats_file = mt4_files_path + f"Indicator_Stats_{symbol}.csv"

print("--- START ANALYSE ---")

try:
    stats_df = pd.read_csv(stats_file, sep=';')
    df = pd.read_csv(data_file, sep=';')
    df = df.dropna(axis=1, how='all')
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Split op 2024
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

    # --- AUTOMATISCHE REDUNDANTIE FILTER ---
    # Alleen kolommen gebruiken die de eerdere filters hebben overleefd
    if USE_REDUNDANTIE_FILTER:
        print(">> REDUNDANTIE FILTER")
        available_cols = [c for c in X.columns if c in final_params['Name'].values]
        corr_matrix = train_df[available_cols].corr().abs()
        
        to_drop = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.90:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    
                    # Veilig ophalen van weights
                    w_i_series = final_params.loc[final_params['Name'] == col_i, 'Weight']
                    w_j_series = final_params.loc[final_params['Name'] == col_j, 'Weight']
                    
                    if not w_i_series.empty and not w_j_series.empty:
                        if w_i_series.values[0] > w_j_series.values[0]:
                            to_drop.append(col_j)
                        else:
                            to_drop.append(col_i)
    
        # Verwijder de dubbele indicatoren
        final_params = final_params[~final_params['Name'].isin(list(set(to_drop)))]
        print(f"ℹ️ Wiskundig filter: {len(set(to_drop))} redundante indicatoren automatisch verwijderd.")

    # Timeframe Penalty/Boost (Alleen als switch aan staat)
    if USE_TIMEFRAME_ADJUSTMENT:
        print(">> TIMEFRAME_ADJUSTMENT")
        def adjust_weight(row):
            w = row['Weight']
            if 'H4' in row['Name']: return w * 0.5
            if 'H1' in row['Name']: return w * 0.7
            if 'M5' in row['Name']: return w * 0.8
            return w
        final_params['Weight'] = final_params.apply(adjust_weight, axis=1)
        print("ℹ️ Timeframe adjustments toegepast.")
    else:
        print("ℹ️ Geen Timeframe adjustments toegepast.")
    
    # Selectie van de Top 15
    if USE_FAMILY_SELECTION:
        print(">> FAMILY_SELECTION")
        # Verdeeld over families (6 Osc, 7 Trend, 2 Vol)
        top_osc   = final_params[final_params['Family'] == 0].sort_values('Weight', ascending=False).head(6)
        top_trend = final_params[final_params['Family'] == 1].sort_values('Weight', ascending=False).head(6)
        top_vol   = final_params[final_params['Family'] == 2].sort_values('Weight', ascending=False).head(3)
        diverse_top_15 = pd.concat([top_osc, top_trend, top_vol])
        print("ℹ️ Selectie: Top per Family toegepast.")
    else:
        # Puur de 15 met de hoogste Weight, ongeacht familie of timeframe
        diverse_top_15 = final_params.sort_values('Weight', ascending=False).head(15)
        print("ℹ️ Selectie: Puur op basis van hoogste Weight (Top 15).")

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
    
    # --- Bereken Threshold ---
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
#
#    # Bepaal de threshold waarde (95e percentiel = top 5%)
#    # Gebruik absolute waarden als trades beide kanten op kunnen, 
#    # of percentiel 95 als het om directionele score gaat. 
#    # Hier gaan we uit van de 'Long' trigger (95th percentile) als generieke threshold.
#    calculated_threshold = np.percentile(scores, 95)
#    
#    print(f"Berekende Threshold (95%): {calculated_threshold:.5f}")


    # --- DYNAMISCHE THRESHOLD BEREKENING ---
    delta = train_df['Time'].max() - train_df['Time'].min()
    total_days = delta.days
    total_needed_signals = total_days * DESIRED_SIGNALS_PER_DAY
    
    # Bereken welk percentiel dit is t.o.v. het totaal aantal regels
    # Formule: (1 - (benodigde_signalen / totaal_regels)) * 100
    dynamic_percentile = (1 - (total_needed_signals / len(train_df))) * 100
    
    # Zorg dat het percentiel binnen realistische grenzen blijft (bijv. max 99.9)
    dynamic_percentile = min(max(dynamic_percentile, 90.0), 99.9)
    
    calculated_threshold = np.percentile(scores, dynamic_percentile)
    
    print(f">> DYNAMISCHE ANALYSE")
    print(f"Periode: {total_days} dagen. Doel: {DESIRED_SIGNALS_PER_DAY} signaal/dag.")
    print(f"Berekend Percentiel: {dynamic_percentile:.2f}%")
    print(f"Berekende Threshold: {calculated_threshold:.5f}")



    # --- ANALYSE SIGNAALKWALITEIT ---
    # Statistische Signaalkwaliteit:
    # StdDev (Standaarddeviatie): Meet de spreiding van de totaalscore. Een hogere waarde duidt op een krachtigere spreiding tussen 'ruis' en 'signaal'.
    # Signaal-Ruis Verhouding:    Dit getal geeft aan hoeveel 'sigma' (standaarddeviaties) de threshold verwijderd is van het gemiddelde.
    #    Waarde > 1.6: De 5% beste signalen wijken duidelijk af van de rest. Dit wijst op een robuuste combinatie waarbij indicatoren elkaar versterken.
    #    Waarde < 1.0: De signalen liggen te dicht bij de ruis. De gekozen indicatoren spreken elkaar tegen of voegen te weinig unieke waarde toe.
    # Doel: Hoe hoger de SNR, hoe groter de kans dat een trade gebaseerd is op een statistisch uniek moment in de markt in plaats van toevallige samenloop van ruis.

    print(">> ANALYSE SIGNAALKWALITEIT")
    score_std = np.std(scores)
    score_mean = np.mean(scores)
    signal_to_noise = calculated_threshold / score_std if score_std != 0 else 0
    
    print(f"Score Statistieken: Mean={score_mean:.5f}, StdDev={score_std:.5f}")
    print(f"Signaal-Ruis Verhouding: {signal_to_noise:.2f}x de standaarddeviatie")

    # --- PERCENTAGE OF AGREEMENT (CONFLUENCE CHECK) ---
    # We kijken naar de momenten waarop een trade wordt getriggerd (score >= threshold)
    print(">> CONFLUENCE CHECK")
    trade_indices = np.where(scores >= calculated_threshold)[0]
    
    if len(trade_indices) > 0:
        agreement_counts = np.zeros(len(trade_indices))
        
        for idx, row in diverse_top_15.iterrows():
            col_name = row['Name']
            if col_name in train_df.columns:
                # Bereken individuele bijdrage (Z-score * direction)
                values = train_df[col_name].values[trade_indices]
                m, s, d = row['Mean'], row['StdDev'], row['Direction']
                if s == 0: s = 1
                
                # Indicator is positief als hij de richting van de trade steunt
                contribution = ((values - m) / s) * d
                agreement_counts += (contribution > 0).astype(int)
        
        # Bereken gemiddelde agreement percentage
        avg_agreement = (agreement_counts.mean() / len(diverse_top_15)) * 100
        print(f"Gemiddelde Agreement bij trades: {avg_agreement:.1f}%")
        
        # Uitleg:
        # > 75%: Sterke confluentie. De indicatoren vormen een unaniem front.
        # < 60%: Zwakke confluentie. De score wordt gedragen door enkele uitschieters; risico op ruis.
    else:
        print("Geen trades gevonden om agreement te berekenen.")


    # --- ESTIMATED SIGNAL FREQUENCY ---
    # Bereken het aantal jaren in de trainingsdata
    print(">> ESTIMATED SIGNAL FREQUENCY")
    delta = train_df['Time'].max() - train_df['Time'].min()
    total_years = delta.days / 365.25
    
    if total_years > 0:
        total_signal_moments = len(trade_indices)
        signals_per_year = total_signal_moments / total_years
        
        print(f"Analyse periode: {total_years:.2f} jaar")
        print(f"Verwachte signaal-momenten per jaar: {signals_per_year:.0f}")
        
        # Toelichting:
        # Dit zijn het aantal kaarsen (bars) waarop de score boven de threshold komt.
        # In de praktijk zal je EA dit clusteren tot één trade per signaal-golf.

    # Voeg threshold toe aan dataframe (dezelfde waarde voor elke regel)
    diverse_top_15['Threshold'] = calculated_threshold
    
    # EXPORTEER
    output_columns = ['ID', 'Name', 'Weight', 'Mean', 'StdDev', 'Direction', 'Threshold']
    diverse_top_15[output_columns].to_csv(tester_files_path + "JJ_Scoring_Params.csv", sep=';', index=False)
    
    print(f"✅ SUCCES: Alleen de 15 top-indicatoren + Threshold zijn opgeslagen in {tester_files_path}JJ_Scoring_Params.csv")
    
    print(diverse_top_15[['ID', 'Name', 'Weight', 'Family']])

except Exception as e:
    print(f"Er ging iets mis: {e}")