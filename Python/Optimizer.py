import subprocess
import itertools
import os
import datetime
import time
import psutil

# Stel de prioriteit van het huidige proces in
proc = psutil.Process(os.getpid())
proc.nice(psutil.HIGH_PRIORITY_CLASS)

# ============================================================================
# LOGGING SETUP
# ============================================================================
# Maak een unieke bestandsnaam op basis van starttijd
current_start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"Optimization_Log_{current_start_time_str}.txt"

def log_print(message):
    """Print naar console én schrijf naar bestand."""
    print(message)
    try:
        # 'a' voor append (toevoegen) zodat we niets overschrijven
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"⚠️ Kon niet schrijven naar logbestand: {e}")

# ============================================================================
# CONFIGURATIE
# ============================================================================
n_osc_range    = [6, 7, 8]      # Focus op hoge confluentie
n_trend_range  = [2, 3, 4]      # Beperk trend-stapeling
n_vol_range    = [2, 3]
red_thresh_opt = [0.51, 0.52]   # Cruciaal: laat zwakkere oscillators toe voor confluentie
corr_h4_range  = [0.8, 1.0, 1.2]
corr_h1_range  = [1.0]
corr_m30_range = [1.0]
corr_m5_range  = [1.0, 1.2, 1.5]
tf_adj         = ['True']
fam_sel        = ['True']

# Alle combinaties genereren
combinations = list(itertools.product(
    n_osc_range, n_trend_range, n_vol_range, red_thresh_opt, corr_h4_range, corr_h1_range, corr_m30_range, corr_m5_range, tf_adj, fam_sel
))

best_final_score = -1
best_params = None
total_jobs = len(combinations)
global_start_time = time.time() # Starttijd voor ETA

log_print(f"--- START OPTIMALISATIE RUN ---")
log_print(f"Logbestand: {log_filename}")
log_print(f"Aantal combinaties: {total_jobs}")
log_print("-" * 50)

for i, (osc, trend, vol, red, h4, h1, m30, m5, ta_val, fs_val) in enumerate(combinations):
    # Start tijdmeting voor deze specifieke job
    start_time = time.time()
    
    cmd = [
        "python", "Market_Analyzer.py",
        "--n_osc", str(osc),
        "--n_trend", str(trend),
        "--n_vol", str(vol),
        "--red_threshold", str(red),
        "--corr_h4", str(h4),
        "--corr_h1", str(h1),
        "--corr_m30", str(m30),
        "--corr_m5", str(m5),
        "--use_tf_adj", str(ta_val),
        "--use_fam_sel", str(fs_val)
    ]
    
    # Uitvoeren en resultaat opvangen
    # proc = subprocess.run(cmd, capture_output=True, text=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Eind tijdmeting
    duration = time.time() - start_time

    # ETA Berekening
    elapsed = time.time() - global_start_time
    avg_time = elapsed / (i + 1)
    remaining_jobs = total_jobs - (i + 1)
    eta_seconds = avg_time * remaining_jobs
    
    # Datum en Tijd toevoegen aan ETA
    eta_datetime = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
    eta_time_str = eta_datetime.strftime('%Y-%m-%d %H:%M:%S')

    try:
        #current_score = float(proc.stdout.strip())
        current_score = float(res.stdout.strip())
        
        # Check voor nieuwe high score
        is_new_best = False
        if current_score > best_final_score:
            best_final_score = current_score
            best_params = {
                "N_OSC": osc, "N_TREND": trend, "N_VOL": vol, "REDUNDANCY": red, 
                "CORR_H4": h4, "CORR_H1": h1, "CORR_M30": m30, "CORR_M5": m5,
                "TF_ADJ": ta_val, "FAM_SEL": fs_val
            }
            is_new_best = True
        
        curr_params = { 
            "OSC": osc, "TRND": trend, "VOL": vol, "RED": red,
            "H4": h4, "H1": h1, "M30": m30, "M5": m5, 
            "TF": ta_val, "FS": fs_val 
        }
        
        # Formatteer output string
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        status_symbol = "🌟 NEW BEST" if is_new_best else ""
        
        log_msg = (f"[{timestamp}] Progress: {i+1}/{total_jobs} | "
                   f"Score: {current_score:.4f} (Best: {best_final_score:.4f}) | "
                   f"{curr_params} | {duration:.2f}s | ETA: {eta_time_str} {status_symbol}")
        
        log_print(log_msg)
            
    except ValueError:
        log_print(f"⚠️ Fout bij lezen output job {i+1}: {res.stdout.strip()}")
        continue

log_print("\n" + "="*30)
log_print(f"OPTIMALISATIE VOLTOOID")
log_print(f"Beste Score: {best_final_score}")
log_print(f"Beste Parameters: {best_params}")
log_print("="*30)