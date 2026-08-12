import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json, warnings, os
import gspread
from google.oauth2.service_account import Credentials
warnings.filterwarnings('ignore')

SHEET_ID   = '1uXWcJ2zknKRuXpqj6dlYPgaolvKc-7G4MM7kPFEMEdU'
SHEET_NAME = 'Proposals-awards Fy22-Fy26'
OUTCOME_COL = 'Year Awarded (P: pending, D: declined, AC: Accepted pre-proposal, U: under consideration)'
NONCOMPETITIVE = {'Continuation','Amendment','Modification',
                  'No-cost Ext','Supplement','Transfer','Renewal'}

# ── 1. Load from Google Sheets ─────────────────────────────────────────────────
def load_sheet():
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    if creds_json:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(creds_json)
            creds_path = f.name
    else:
        creds_path = 'credentials.json'

    scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    creds  = Credentials.from_service_account_file(creds_path, scopes=scopes)
    client = gspread.authorize(creds)
    sheet  = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
    data   = sheet.get_all_records()
    df     = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows from Google Sheets")
    return df

def group_sponsor(s):
    s = str(s).upper().strip()
    if any(x in s for x in ['MTU','ICC','GLRC']):          return 'MTU_Internal'
    if 'NSF' in s:                                          return 'NSF'
    if any(x in s for x in ['NIH','USDHHS','NIOSH','CDC']): return 'NIH_Health'
    if any(x in s for x in ['DOD','DARPA','NAVY','ARMY','NSWC','NRL','ONR']): return 'DOD'
    if 'DOE' in s:                                          return 'DOE'
    if 'NASA' in s:                                         return 'NASA'
    if any(x in s for x in ['NIST','ORAU','ARPA','FCAT']): return 'Other_Federal'
    if any(x in s for x in ['AMAZON','GOOGLE','MICROSOFT','SAMSUNG','SONY','CISCO','TOYOTA']): return 'Industry'
    if any(x in s for x in ['FOUNDATION','AHA','BCBS','CLIMATE','SUPERIOR','HEALTH']): return 'NonProfit'
    if any(x in s for x in ['MDOT','MIDOT','WIDOT','MNDOT','USDOT']): return 'Transportation'
    return 'Other'

def group_funding(f):
    f = str(f).strip().upper()
    if f in ('FEDERAL','FPT'): return 'Federal'
    if f == 'INTERNAL':        return 'Internal'
    if f == 'INDUSTRY':        return 'Industry'
    if f == 'NON-PROFIT':      return 'NonProfit'
    if f == 'STATE':           return 'State'
    return 'Other'

# ── 2. Load & engineer features ────────────────────────────────────────────────
df = load_sheet()

for col in df.select_dtypes('object').columns:
    df[col] = df[col].astype(str).str.strip()
df[OUTCOME_COL]  = df[OUTCOME_COL].str.strip()
df['Department'] = df['Department'].str.strip()
df['sponsor_group'] = df['Sponsor'].apply(group_sponsor)
df['funding_group']  = df['Funding Type'].apply(group_funding)
df['is_competitive'] = (~df['Type'].isin(NONCOMPETITIVE)).astype(int)
df['Total proposal amount'] = pd.to_numeric(df['Total proposal amount'], errors='coerce')
df['Duration (years)']      = pd.to_numeric(df['Duration (years)'],      errors='coerce')
df['Fiscal Year Submitted']  = pd.to_numeric(df['Fiscal Year Submitted'], errors='coerce')
df['log_amount'] = np.log1p(df['Total proposal amount'].fillna(0))

# ── 3. Known outcomes + leakage-safe PI features ───────────────────────────────
known = df[df[OUTCOME_COL].isin(['D','22','23','24','25','26'])].copy()
known['awarded'] = (known[OUTCOME_COL] != 'D').astype(int)
known = known.sort_values('Fiscal Year Submitted').reset_index(drop=True)

pi_stats, pi_sp_hist = {}, {}
pi_prior_rate, pi_sp_rate = [], []

for _, row in known.iterrows():
    pi  = str(row.get('PI Name ', row.get('PI Name',''))).strip()
    sp  = row['sponsor_group']
    s   = pi_stats.get(pi, {'w':0,'n':0})
    ks  = pi_sp_hist.get((pi,sp), {'w':0,'n':0})
    pi_prior_rate.append(s['w']/s['n'] if s['n']>0 else np.nan)
    pi_sp_rate.append(ks['w']/ks['n'] if ks['n']>0 else np.nan)
    pi_stats[pi]       = {'w':s['w']+row['awarded'], 'n':s['n']+1}
    pi_sp_hist[(pi,sp)] = {'w':ks['w']+row['awarded'],'n':ks['n']+1}

overall_rate = known['awarded'].mean()
known['pi_prior_rate']   = pd.Series(pi_prior_rate).fillna(overall_rate).values
known['pi_sponsor_rate'] = pd.Series(pi_sp_rate).fillna(overall_rate).values

# ── 4. Train model ─────────────────────────────────────────────────────────────
comp = known[known['is_competitive']==1].copy()
features_cat = ['sponsor_group','funding_group','Department']
features_num = ['pi_prior_rate','pi_sponsor_rate','log_amount',
                'Duration (years)','Fiscal Year Submitted']

X_cat = pd.get_dummies(comp[features_cat], drop_first=False)
X_num = comp[features_num].fillna(0)
X     = pd.concat([X_num, X_cat], axis=1).astype(float)
y     = comp['awarded']

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(max_iter=1000, random_state=42,
                                  C=0.5, class_weight='balanced'))
])
pipe.fit(X, y)
print(f"Model trained on {len(X)} proposals, {len(X.columns)} features")
print(f"Overall success rate: {overall_rate:.1%}")

# ── 5. Export model_export.json ────────────────────────────────────────────────
from datetime import datetime
scaler = pipe.named_steps['scaler']
model  = pipe.named_steps['model']

model_export = {
    'intercept':      float(model.intercept_[0]),
    'feature_names':  list(X.columns),
    'coefficients':   [float(c) for c in model.coef_[0]],
    'scaler_mean':    [float(m) for m in scaler.mean_],
    'scaler_std':     [float(s) for s in scaler.scale_],
    'overall_rate':   float(overall_rate),
    'threshold':      0.51,
    'trained_on':     len(X),
    'last_updated':   datetime.utcnow().strftime('%Y-%m-%d'),
}

with open('model_export.json', 'w') as f:
    json.dump(model_export, f, indent=2)
print(f"Saved → model_export.json  (trained on {len(X)} proposals)")
print("Done.")
