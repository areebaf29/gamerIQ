# =============================================================================
# app.py — GamerIQ (Light gaming UI + LR vs RF comparison)
# =============================================================================
import sqlite3, numpy as np, pandas as pd, os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET","POST","OPTIONS"])
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gamer.db")

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        res = Response()
        res.headers["Access-Control-Allow-Origin"]  = "*"
        res.headers["Access-Control-Allow-Headers"] = "Content-Type"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return res

# =============================================================================
# Helpers
# =============================================================================
def calc_productivity_drop(category, playtime, sleep):
    if category == "Professional": return 0
    return int(min(100, round(max(0, playtime-3)*8 + ((6-sleep)*7 if sleep<6 else 0))))

def addiction_label(playtime, sleep, rage, prod):
    if playtime >= 7:               return "High"
    if playtime >= 5 and sleep < 6: return "High"
    if playtime >= 5 and rage >= 7: return "High"
    if prod >= 50:                  return "High"
    return "Low"

def cluster_label(playtime):
    if playtime < 4:  return "Low Engagement"
    if playtime <= 8: return "Balanced"
    return "High Engagement"

FEATURES = ["playtime","sleep","gaming_days","rage_level","productivity_drop"]

# =============================================================================
# Generate dataset
# =============================================================================
def generate_dataset(n=2000):
    np.random.seed(42)
    pt = np.random.uniform(0.5, 16, n)
    sl = np.random.uniform(3, 12, n)
    gd = np.random.randint(1, 8, n)
    rl = np.random.randint(1, 11, n)
    ok = (pt + sl) <= 24
    pt,sl,gd,rl = pt[ok],sl[ok],gd[ok],rl[ok]
    n  = len(pt)
    pd_ = np.array([calc_productivity_drop("Casual",pt[i],sl[i]) for i in range(n)])
    # Add slight noise to avoid 100% accuracy
    noise = np.random.rand(n) < 0.06
    add_base = np.array([addiction_label(pt[i],sl[i],rl[i],pd_[i]) for i in range(n)])
    add = np.where(noise, np.where(add_base=="High","Low","High"), add_base)
    ply = np.where(pt<3,"Casual",np.where(pt<=6,"Competitive","Hardcore"))
    return pd.DataFrame({"playtime":pt,"sleep":sl,"gaming_days":gd,"rage_level":rl,
                         "productivity_drop":pd_,"addiction":add,"player_type":ply})

# =============================================================================
# Train models
# =============================================================================
def train_models():
    df  = generate_dataset()
    X   = df[FEATURES]
    le_add = LabelEncoder()
    y_add  = le_add.fit_transform(df["addiction"])
    Xtr,Xte,ytr,yte = train_test_split(X,y_add,test_size=0.2,random_state=42)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, C=0.8)
    lr.fit(Xtr_s, ytr)
    lr_acc = float(lr.score(Xte_s, yte))
    print(f"[ML] Logistic Regression : {lr_acc:.2%}")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=40, max_depth=4, max_features='sqrt',
                                 min_samples_leaf=8, random_state=7)
    rf.fit(Xtr, ytr)
    rf_acc = float(rf.score(Xte, yte))
    print(f"[ML] Random Forest       : {rf_acc:.2%}")

    # --- Decision Tree (player type) ---
    le_ply = LabelEncoder()
    y_ply  = le_ply.fit_transform(df["player_type"])
    Xtr2,Xte2,ytr2,yte2 = train_test_split(X,y_ply,test_size=0.2,random_state=42)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(Xtr2, ytr2)
    dt_acc = float(dt.score(Xte2, yte2))
    print(f"[ML] Decision Tree       : {dt_acc:.2%}")

    # --- K-Means sample ---
    idx = [int(i) for i in np.random.choice(len(df),min(300,len(df)),replace=False)]

    lr_info = {
        "accuracy":   round(lr_acc*100,2),
        "train_size": int(len(Xtr)), "test_size": int(len(Xte)),
        "high_count": int((df["addiction"]=="High").sum()),
        "low_count":  int((df["addiction"]=="Low").sum()),
        "coefficients":{f:round(float(c),4) for f,c in zip(FEATURES,lr.coef_[0])},
    }
    rf_info = {
        "accuracy":   round(rf_acc*100,2),
        "feature_importance":{f:round(float(v),4) for f,v in zip(FEATURES,rf.feature_importances_)},
    }
    dt_info = {
        "accuracy":   round(dt_acc*100,2),
        "train_size": int(len(Xtr2)), "test_size": int(len(Xte2)),
        "max_depth":  int(dt.get_depth()), "n_leaves": int(dt.get_n_leaves()),
        "feature_importance":{f:round(float(v),4) for f,v in zip(FEATURES,dt.feature_importances_)},
        "class_counts":{cls:int((df["player_type"]==cls).sum()) for cls in le_ply.classes_},
    }
    km_info = {
        "scatter":[{"playtime":round(float(df["playtime"].iloc[i]),2),
                    "sleep":round(float(df["sleep"].iloc[i]),2),
                    "prod":round(float(df["productivity_drop"].iloc[i]),1),
                    "cluster":cluster_label(float(df["playtime"].iloc[i]))} for i in idx],
    }
    return lr,le_add,sc,rf,dt,le_ply,lr_info,rf_info,dt_info,km_info

print("\n[ML] Training models...")
LR,LE_ADD,SCALER,RF,DT,LE_PLY,LR_INFO,RF_INFO,DT_INFO,KM_INFO = train_models()
N_ROWS = LR_INFO["train_size"] + LR_INFO["test_size"]  # rows actually used for training/testing
print("[ML] All models ready.\n")

# =============================================================================
# Database
# =============================================================================
def get_db():
    conn = sqlite3.connect(DB_PATH); conn.row_factory=sqlite3.Row; return conn

def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS gamer_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,age INTEGER,gender TEXT,player_category TEXT,
        playtime REAL,sleep REAL,gaming_days INTEGER,game_type TEXT,
        rage_level INTEGER,productivity_drop REAL,
        addiction_lr TEXT,addiction_rf TEXT,player_type TEXT,cluster TEXT,
        created TEXT DEFAULT(datetime('now')))""")
    conn.commit(); conn.close()

init_db()

# =============================================================================
# API
# =============================================================================
@app.route("/predict",methods=["POST","OPTIONS"])
def predict():
    data = request.get_json()
    if not data: return jsonify({"error":"No JSON body"}),400
    req = ["name","age","gender","player_category","playtime","sleep","gaming_days","game_type","rage_level"]
    missing = [f for f in req if f not in data]
    if missing: return jsonify({"error":f"Missing:{missing}"}),400
    pt,sl = data["playtime"],data["sleep"]
    if not(0.5<=pt<=16): return jsonify({"error":"Playtime must be 0.5–16 hrs"}),400
    if not(3<=sl<=12):   return jsonify({"error":"Sleep must be 3–12 hrs"}),400
    if pt+sl>24:         return jsonify({"error":"Playtime+sleep cannot exceed 24h"}),400
    if not(1<=data["rage_level"]<=10): return jsonify({"error":"Rage must be 1–10"}),400
    if not(1<=data["gaming_days"]<=7): return jsonify({"error":"Gaming days must be 1–7"}),400

    prod = calc_productivity_drop(data["player_category"],pt,sl)
    X    = pd.DataFrame([{f:(data[f] if f!="productivity_drop" else prod) for f in FEATURES}])[FEATURES]
    Xs   = SCALER.transform(X)

    add_lr  = LE_ADD.inverse_transform(LR.predict(Xs))[0]
    add_rf  = LE_ADD.inverse_transform(RF.predict(X))[0]
    ply     = LE_PLY.inverse_transform(DT.predict(X))[0]
    cluster = cluster_label(pt)

    conn=get_db()
    conn.execute("""INSERT INTO gamer_data
        (name,age,gender,player_category,playtime,sleep,gaming_days,game_type,
         rage_level,productivity_drop,addiction_lr,addiction_rf,player_type,cluster)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (data["name"],data["age"],data["gender"],data["player_category"],
         pt,sl,data["gaming_days"],data["game_type"],data["rage_level"],
         prod,add_lr,add_rf,ply,cluster))
    conn.commit(); conn.close()

    return jsonify({"addiction_lr":add_lr,"addiction_rf":add_rf,
                    "player_type":ply,"cluster":cluster,"productivity_drop":prod,
                    "models_agree": add_lr==add_rf})

@app.route("/records",methods=["GET","OPTIONS"])
def records():
    conn=get_db()
    rows=conn.execute("SELECT * FROM gamer_data ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/model-info",methods=["GET","OPTIONS"])
def model_info():
    return jsonify({"lr":LR_INFO,"rf":RF_INFO,"dt":DT_INFO,"km":KM_INFO})

# =============================================================================
# Shared styles — light gaming theme
# =============================================================================
GAME_BG = """
  background-color: #f0f2f5;
  background-image:
    radial-gradient(circle at 15% 25%, rgba(99,102,241,0.08) 0%, transparent 50%),
    radial-gradient(circle at 85% 75%, rgba(236,72,153,0.08) 0%, transparent 50%),
    url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%236366f1' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
"""

SHARED_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;500;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',system-ui,sans-serif;{GAME_BG}color:#1e1b4b;min-height:100vh;}}
.con{{max-width:1100px;margin:0 auto;padding:2rem 1.5rem;}}
.pt{{font-size:28px;font-weight:700;font-family:'Orbitron',sans-serif;
     background:linear-gradient(135deg,#6366f1,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;}}
.ps{{font-size:13px;color:#6b7280;margin-bottom:1.5rem;}}
.card{{background:rgba(255,255,255,0.9);border:1.5px solid rgba(99,102,241,0.15);border-radius:16px;padding:1.5rem;backdrop-filter:blur(8px);box-shadow:0 4px 20px rgba(99,102,241,0.08);}}
.ct{{font-size:11px;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:.08em;margin-bottom:1rem;}}
.field{{margin-bottom:12px;}}
label{{font-size:12px;color:#6b7280;display:block;margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;}}
input,select{{width:100%;padding:10px 14px;background:rgba(248,249,255,0.9);border:1.5px solid #e0e7ff;border-radius:10px;color:#1e1b4b;font-size:14px;font-family:'Inter',sans-serif;outline:none;transition:all .2s;}}
input:focus,select:focus{{border-color:#6366f1;background:white;box-shadow:0 0 0 3px rgba(99,102,241,0.12);}}
select option{{background:white;color:#1e1b4b;}}
.r2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
.hint{{font-size:11px;color:#9ca3af;margin-top:3px;}}
.hint.w{{color:#ef4444;}}
.alert{{background:#fef2f2;border:1.5px solid #fecaca;border-radius:10px;padding:10px 14px;font-size:13px;color:#dc2626;margin-bottom:12px;display:none;line-height:1.6;}}
.sbtn{{width:100%;padding:13px;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;border:none;border-radius:12px;font-size:15px;font-weight:700;font-family:'Orbitron',sans-serif;cursor:pointer;margin-top:10px;letter-spacing:.05em;transition:all .2s;box-shadow:0 4px 15px rgba(99,102,241,0.35);}}
.sbtn:hover{{transform:translateY(-1px);box-shadow:0 6px 20px rgba(99,102,241,0.45);}}
.sbtn:disabled{{background:#d1d5db;cursor:not-allowed;box-shadow:none;transform:none;}}
.badge{{display:inline-block;font-size:11px;font-weight:700;padding:4px 12px;border-radius:20px;letter-spacing:.03em;}}
.bh{{background:#fee2e2;color:#dc2626;border:1px solid #fca5a5;}}
.bl{{background:#dcfce7;color:#16a34a;border:1px solid #86efac;}}
.bc{{background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;}}
.bco{{background:#fef3c7;color:#b45309;border:1px solid #fcd34d;}}
.bhc{{background:#fce7f3;color:#be185d;border:1px solid #f9a8d4;}}
.ble{{background:#f3f4f6;color:#6b7280;border:1px solid #d1d5db;}}
.bba{{background:#d1fae5;color:#065f46;border:1px solid #6ee7b7;}}
.bhe{{background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd;}}
"""

NAV_CSS = """
nav{background:rgba(255,255,255,0.95);padding:0 2rem;display:flex;align-items:center;
    justify-content:space-between;height:60px;position:sticky;top:0;z-index:100;
    border-bottom:2px solid rgba(99,102,241,0.15);backdrop-filter:blur(12px);
    box-shadow:0 2px 20px rgba(99,102,241,0.1);}
.nav-logo{font-family:'Orbitron',sans-serif;font-weight:900;font-size:18px;
          background:linear-gradient(135deg,#6366f1,#ec4899);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.nav-links{display:flex;gap:4px;}
.nav-link{padding:7px 18px;border-radius:8px;font-size:13px;font-weight:600;
           text-decoration:none;transition:all .2s;color:#6b7280;}
.nav-link:hover{background:#f0f0ff;color:#6366f1;}
.nav-link.active{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;
                  box-shadow:0 3px 10px rgba(99,102,241,0.3);}
"""

def nav(active):
    pages = [('predict','/','Predict'),('history','/history','History'),('models','/models','ML Models')]
    links = ''.join(f'<a href="{href}" class="nav-link{" active" if p==active else ""}">{label}</a>'
                    for p,href,label in pages)
    return f'<nav><span class="nav-logo">🎮 GamerIQ</span><div class="nav-links">{links}</div></nav>'

# =============================================================================
# Predict page
# =============================================================================
@app.route("/")
def page_predict():
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GamerIQ — Predict</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>{SHARED_CSS}{NAV_CSS}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;align-items:start;}}
.cat-banner{{border-radius:10px;padding:10px 14px;font-size:12px;margin-bottom:12px;display:none;font-weight:500;}}
.cat-cas{{background:#eff6ff;border:1.5px solid #bfdbfe;color:#1d4ed8;}}
.cat-pro{{background:#f0fdf4;border:1.5px solid #bbf7d0;color:#15803d;}}
.prod-box{{background:linear-gradient(135deg,#f0f0ff,#fdf4ff);border:1.5px solid #c4b5fd;border-radius:10px;padding:10px 14px;font-size:18px;color:#6366f1;font-weight:700;font-family:'Orbitron',sans-serif;}}
.prod-sub{{font-size:11px;color:#9ca3af;margin-top:4px;}}
.rc{{background:linear-gradient(135deg,rgba(99,102,241,0.05),rgba(236,72,153,0.05));
    border:1.5px solid rgba(99,102,241,0.2);border-radius:16px;padding:1.5rem;margin-top:1rem;display:none;}}
.rr{{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid rgba(99,102,241,0.1);}}
.rr:last-child{{border-bottom:none;}}
.rl{{font-size:13px;color:#6b7280;font-weight:500;}}
.agree-box{{background:#ecfdf5;border:1.5px solid #6ee7b7;border-radius:10px;padding:8px 14px;font-size:12px;color:#065f46;font-weight:600;margin-top:10px;text-align:center;display:none;}}
.disagree-box{{background:#fff7ed;border:1.5px solid #fed7aa;border-radius:10px;padding:8px 14px;font-size:12px;color:#c2410c;font-weight:600;margin-top:10px;text-align:center;display:none;}}
.info-box{{background:rgba(255,255,255,0.9);border:1.5px solid rgba(99,102,241,0.15);border-radius:16px;padding:1.5rem;backdrop-filter:blur(8px);}}
.ii{{display:flex;gap:12px;margin-bottom:1rem;padding-bottom:1rem;border-bottom:1px solid #f3f4f6;}}
.ii:last-child{{margin:0;padding:0;border:none;}}
.ii-t{{font-size:13px;font-weight:700;color:#1e1b4b;margin-bottom:3px;}}
.ii-d{{font-size:12px;color:#6b7280;line-height:1.55;}}
.tag{{display:inline-block;font-size:10px;padding:2px 8px;border-radius:20px;font-weight:700;margin-top:5px;}}
.tlr{{background:#ede9fe;color:#5b21b6;}}.trf{{background:#fce7f3;color:#be185d;}}.tdt{{background:#dcfce7;color:#15803d;}}.tkm{{background:#fef3c7;color:#b45309;}}
.total-hint{{font-size:12px;margin-bottom:10px;}}
</style></head><body>
{nav('predict')}
<div class="con">
  <p class="pt">PREDICT</p>
  <p class="ps">Enter your gaming profile — LR &amp; Random Forest will both analyse your addiction risk</p>
  <div class="grid">
    <div>
      <div class="card">
        <p class="ct">⚡ Personal Info</p>
        <div class="r2">
          <div class="field"><label>Full name</label><input id="name" placeholder="e.g. Saniya"/></div>
          <div class="field"><label>Age</label><input id="age" type="number" min="10" max="60" placeholder="22"/></div>
        </div>
        <div class="r2">
          <div class="field"><label>Gender</label>
            <select id="gender"><option value="">Select</option><option>Male</option><option>Female</option><option>Other</option></select></div>
          <div class="field"><label>Player Category</label>
            <select id="player_category" onchange="onCat()"><option value="">Select</option><option value="Casual">Casual</option><option value="Professional">Professional</option></select></div>
        </div>
        <div id="catBanner" class="cat-banner"></div>
        <p class="ct" style="margin-top:.75rem;">🎮 Gaming Habits</p>
        <div id="alertBox" class="alert"></div>
        <div class="r2">
          <div class="field"><label>Playtime (hrs/day)</label>
            <input id="playtime" type="number" min="0.5" max="16" step="0.5" placeholder="7" oninput="onHours()"/>
            <p class="hint" id="ph">0.5 – 16 hrs</p></div>
          <div class="field"><label>Sleep (hrs/day)</label>
            <input id="sleep" type="number" min="3" max="12" step="0.5" placeholder="6" oninput="onHours()"/>
            <p class="hint" id="sh">3 – 12 hrs</p></div>
        </div>
        <p class="hint total-hint" id="totalHint"></p>
        <div class="r2">
          <div class="field"><label>Gaming Days/Week</label><input id="gaming_days" type="number" min="1" max="7" placeholder="5"/></div>
          <div class="field"><label>Rage Level (1–10)</label><input id="rage_level" type="number" min="1" max="10" placeholder="6"/></div>
        </div>
        <div class="field"><label>Game Type</label>
          <select id="game_type"><option>FPS</option><option>RPG</option><option>MOBA</option><option>Sports</option><option>Puzzle</option><option>Battle Royale</option></select></div>
        <div class="field"><label>Productivity Drop (auto)</label>
          <div class="prod-box" id="prodBox">—</div>
          <p class="prod-sub" id="prodSub"></p></div>
        <button class="sbtn" id="submitBtn" onclick="doSubmit()">▶ RUN PREDICTION</button>
      </div>
      <div class="rc" id="resultCard">
        <p class="ct" style="margin-bottom:.75rem;">🏆 Results</p>
        <div class="rr"><span class="rl">🧠 Addiction (Log. Regression)</span><span id="r-lr" class="badge"></span></div>
        <div class="rr"><span class="rl">🌲 Addiction (Random Forest)</span><span id="r-rf" class="badge"></span></div>
        <div class="rr"><span class="rl">🎯 Player Type</span><span id="r-ply" class="badge"></span></div>
        <div class="rr"><span class="rl">📊 Cluster</span><span id="r-cls" class="badge"></span></div>
        <div class="rr"><span class="rl">📉 Productivity Drop</span><span id="r-prod" class="badge bhe"></span></div>
        <div id="agreeBox" class="agree-box">✅ Both models agree on the addiction prediction!</div>
        <div id="disagreeBox" class="disagree-box">⚠️ Models disagree — LR and RF gave different predictions. Check the ML Models page for details.</div>
      </div>
    </div>
    <div class="info-box">
      <p class="ct">🔬 How It Works</p>
      <div class="ii"><div style="font-size:20px;flex-shrink:0;margin-top:2px">🧠</div><div>
        <p class="ii-t">Logistic Regression — Addiction</p>
        <p class="ii-d">Learns a weight per feature. Uses sigmoid to output probability. Fast and interpretable.</p>
        <span class="tag tlr">{LR_INFO["accuracy"]}% test accuracy</span></div></div>
      <div class="ii"><div style="font-size:20px;flex-shrink:0;margin-top:2px">🌲</div><div>
        <p class="ii-t">Random Forest — Addiction (comparison)</p>
        <p class="ii-d">Builds 40 decision trees on random subsets and takes a majority vote. More robust to noise.</p>
        <span class="tag trf">{RF_INFO["accuracy"]}% test accuracy</span></div></div>
      <div class="ii"><div style="font-size:20px;flex-shrink:0;margin-top:2px">🎯</div><div>
        <p class="ii-t">Decision Tree — Player Type</p>
        <p class="ii-d">Casual &lt;3h · Competitive 3–6h · Hardcore &gt;6h/day.</p>
        <span class="tag tdt">{DT_INFO["accuracy"]}% test accuracy</span></div></div>
      <div class="ii"><div style="font-size:20px;flex-shrink:0;margin-top:2px">📊</div><div>
        <p class="ii-t">Engagement Cluster (playtime rule)</p>
        <p class="ii-d">Low &lt;4h · Balanced 4–8h · High Engagement &gt;8h/day.</p>
        <span class="tag tkm">3 clusters</span></div></div>
      <div class="ii"><div style="font-size:20px;flex-shrink:0;margin-top:2px">📉</div><div>
        <p class="ii-t">Productivity Drop — Auto</p>
        <p class="ii-d">Casual: (playtime−3)×8% + sleep penalty if &lt;6h. Professional: always 0%.</p></div></div>
      <div class="ii"><div style="font-size:20px;flex-shrink:0;margin-top:2px">📋</div><div>
        <p class="ii-t">Train-Test Split</p>
        <p class="ii-d">{N_ROWS:,} synthetic rows (2,000 generated, rows with playtime + sleep over 24h dropped) → 80% train / 20% test. Models are fitted on the 80% training split only and scored on the 20% test split. 6% of addiction-risk labels are flipped at random as noise; player type is a direct playtime rule.</p></div></div>
    </div>
  </div>
</div>
<script>
function onCat(){{
  const cat=document.getElementById('player_category').value,b=document.getElementById('catBanner');
  if(!cat){{b.style.display='none';return;}}
  b.style.display='block';
  if(cat==='Casual'){{b.className='cat-banner cat-cas';b.textContent='🎮 Casual — gaming for fun. Productivity drop calculated from your habits.';}}
  else{{b.className='cat-banner cat-pro';b.textContent='🏆 Professional — gaming is your career. Productivity drop = 0%.';}}
  updateProd();
}}
function onHours(){{
  const p=parseFloat(document.getElementById('playtime').value)||0,s=parseFloat(document.getElementById('sleep').value)||0;
  const ph=document.getElementById('ph'),sh=document.getElementById('sh'),th=document.getElementById('totalHint');
  ph.className=p>16?'hint w':'hint'; ph.textContent=p>16?'⚠ Max 16 hrs':'0.5–16 hrs';
  sh.className=(s<3||s>12)?'hint w':'hint'; sh.textContent=(s<3||s>12)?'⚠ Must be 3–12 hrs':'3–12 hrs';
  if(p&&s){{const t=p+s;th.style.color=t>24?'#ef4444':'#9ca3af';
    th.textContent=t>24?`⚠ Total ${{t.toFixed(1)}}h exceeds 24h!`:`Total ${{t.toFixed(1)}}h used · ${{(24-t).toFixed(1)}}h remaining`;}}
  updateProd();
}}
function updateProd(){{
  const cat=document.getElementById('player_category').value,p=parseFloat(document.getElementById('playtime').value)||0,s=parseFloat(document.getElementById('sleep').value)||0;
  const box=document.getElementById('prodBox'),sub=document.getElementById('prodSub');
  if(!cat||!p){{box.textContent='—';sub.textContent='';return;}}
  if(cat==='Professional'){{box.textContent='0%';sub.textContent='Professional gamers — no productivity loss.';return;}}
  const pp=Math.max(0,p-3)*8,sp=s>0&&s<6?(6-s)*7:0,drop=Math.min(100,Math.round(pp+sp));
  box.textContent=drop+'%';sub.textContent=`Play ${{Math.round(pp)}}% + Sleep penalty ${{Math.round(sp)}}% = ${{drop}}%`;
}}
function bc(f,v){{
  if(f==='add') return v==='High'?'badge bh':'badge bl';
  if(f==='ply'){{if(v==='Casual')return'badge bc';if(v==='Competitive')return'badge bco';return'badge bhc';}}
  if(f==='cls'){{if(v==='Low Engagement')return'badge ble';if(v==='Balanced')return'badge bba';return'badge bhe';}}
}}
async function doSubmit(){{
  const btn=document.getElementById('submitBtn'),ab=document.getElementById('alertBox');
  const name=document.getElementById('name').value.trim(),age=parseInt(document.getElementById('age').value);
  const gender=document.getElementById('gender').value,cat=document.getElementById('player_category').value;
  const p=parseFloat(document.getElementById('playtime').value),s=parseFloat(document.getElementById('sleep').value);
  const gd=parseInt(document.getElementById('gaming_days').value),rage=parseInt(document.getElementById('rage_level').value);
  const errs=[];
  if(!name) errs.push('Name required');
  if(!age||age<10||age>60) errs.push('Age must be 10–60');
  if(!gender) errs.push('Select gender');
  if(!cat) errs.push('Select player category');
  if(!p||p<0.5||p>16) errs.push('Playtime must be 0.5–16 hrs');
  if(!s||s<3||s>12) errs.push('Sleep must be 3–12 hrs');
  if(p+s>24) errs.push('Playtime + sleep cannot exceed 24h');
  if(!gd||gd<1||gd>7) errs.push('Gaming days must be 1–7');
  if(!rage||rage<1||rage>10) errs.push('Rage must be 1–10');
  if(errs.length){{ab.innerHTML=errs.map(e=>'⚠ '+e).join('<br>');ab.style.display='block';return;}}
  ab.style.display='none';
  btn.textContent='ANALYSING...';btn.disabled=true;
  const pp=Math.max(0,p-3)*8,sp=s<6?(6-s)*7:0,drop=cat==='Professional'?0:Math.min(100,Math.round(pp+sp));
  try{{
    const res=await fetch('/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{name,age,gender,player_category:cat,playtime:p,sleep:s,
        gaming_days:gd,game_type:document.getElementById('game_type').value,rage_level:rage,productivity_drop:drop}})}});
    const d=await res.json();
    if(d.error){{alert('Error: '+d.error);btn.textContent='▶ RUN PREDICTION';btn.disabled=false;return;}}
    document.getElementById('r-lr').textContent=d.addiction_lr;document.getElementById('r-lr').className=bc('add',d.addiction_lr);
    document.getElementById('r-rf').textContent=d.addiction_rf;document.getElementById('r-rf').className=bc('add',d.addiction_rf);
    document.getElementById('r-ply').textContent=d.player_type;document.getElementById('r-ply').className=bc('ply',d.player_type);
    document.getElementById('r-cls').textContent=d.cluster;document.getElementById('r-cls').className=bc('cls',d.cluster);
    document.getElementById('r-prod').textContent=d.productivity_drop+'%';
    document.getElementById('agreeBox').style.display=d.models_agree?'block':'none';
    document.getElementById('disagreeBox').style.display=d.models_agree?'none':'block';
    document.getElementById('resultCard').style.display='block';
    document.getElementById('resultCard').scrollIntoView({{behavior:'smooth'}});
  }}catch(e){{alert('Cannot connect. Make sure python app.py is running.');}}
  btn.textContent='▶ RUN PREDICTION';btn.disabled=false;
}}
</script></body></html>"""

# =============================================================================
# History page
# =============================================================================
@app.route("/history")
def page_history():
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GamerIQ — History</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>{SHARED_CSS}{NAV_CSS}
.stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:1.5rem;}}
.stat{{background:white;border:1.5px solid rgba(99,102,241,0.15);border-radius:14px;padding:1rem 1.25rem;box-shadow:0 2px 10px rgba(99,102,241,0.06);}}
.sl{{font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.04em;margin-bottom:5px;font-weight:600;}}
.sv{{font-size:22px;font-weight:700;font-family:'Orbitron',sans-serif;}}
.sv.p{{color:#6366f1;}}.sv.r{{color:#ef4444;}}.sv.g{{color:#16a34a;}}
.controls{{display:flex;gap:10px;margin-bottom:1rem;flex-wrap:wrap;align-items:center;}}
select.filter{{padding:8px 12px;background:white;border:1.5px solid #e0e7ff;border-radius:10px;color:#374151;font-size:13px;outline:none;font-family:'Inter',sans-serif;}}
.rbtn{{padding:8px 16px;border:1.5px solid #e0e7ff;background:white;border-radius:10px;cursor:pointer;color:#6366f1;font-size:13px;font-weight:600;transition:all .2s;}}
.rbtn:hover{{background:#f0f0ff;border-color:#a5b4fc;}}
.tcard{{background:white;border:1.5px solid rgba(99,102,241,0.15);border-radius:16px;overflow:auto;box-shadow:0 4px 20px rgba(99,102,241,0.08);}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th{{text-align:left;font-size:10px;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:.06em;padding:12px 12px;border-bottom:2px solid #e0e7ff;background:#f8f7ff;white-space:nowrap;}}
td{{padding:10px 12px;border-bottom:1px solid #f3f4f6;color:#374151;vertical-align:middle;white-space:nowrap;}}
tr:last-child td{{border-bottom:none;}} tr:hover td{{background:#fafaff;}}
.av{{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#ec4899);display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:white;margin-right:6px;vertical-align:middle;}}
.empty{{text-align:center;padding:3rem;color:#9ca3af;font-size:14px;}}
.diff-badge{{font-size:10px;padding:2px 6px;border-radius:10px;background:#fff7ed;color:#c2410c;border:1px solid #fed7aa;font-weight:700;margin-left:4px;}}
</style></head><body>
{nav('history')}
<div class="con">
  <p class="pt">HISTORY</p>
  <p class="ps">All predictions — compare LR vs Random Forest outcomes across all users</p>
  <div class="stats">
    <div class="stat"><p class="sl">Total</p><p class="sv p" id="s-total">—</p></div>
    <div class="stat"><p class="sl">High (LR)</p><p class="sv r" id="s-high">—</p></div>
    <div class="stat"><p class="sl">Models disagree</p><p class="sv" style="color:#f59e0b;" id="s-diff">—</p></div>
    <div class="stat"><p class="sl">Avg playtime</p><p class="sv" id="s-play">—</p></div>
    <div class="stat"><p class="sl">Avg sleep</p><p class="sv" id="s-sleep">—</p></div>
  </div>
  <div class="controls">
    <select class="filter" id="fA" onchange="filt()"><option value="">All addiction (LR)</option><option value="High">High</option><option value="Low">Low</option></select>
    <select class="filter" id="fR" onchange="filt()"><option value="">All addiction (RF)</option><option value="High">High</option><option value="Low">Low</option></select>
    <select class="filter" id="fP" onchange="filt()"><option value="">All player types</option><option value="Casual">Casual</option><option value="Competitive">Competitive</option><option value="Hardcore">Hardcore</option></select>
    <select class="filter" id="fC" onchange="filt()"><option value="">All clusters</option><option value="Low Engagement">Low Engagement</option><option value="Balanced">Balanced</option><option value="High Engagement">High Engagement</option></select>
    <button class="rbtn" onclick="load()">↻ Refresh</button>
  </div>
  <div class="tcard">
    <table><thead><tr>
      <th>#</th><th>Player</th><th>Age</th><th>Category</th><th>Game</th>
      <th>Play</th><th>Sleep</th><th>Rage</th><th>Prod.Drop</th>
      <th>LR Addiction</th><th>RF Addiction</th><th>Match?</th>
      <th>Type</th><th>Cluster</th><th>Date</th>
    </tr></thead><tbody id="tbody"><tr><td colspan="15" class="empty">Loading...</td></tr></tbody></table>
  </div>
</div>
<script>
let all=[];
function badge(f,v){{
  if(f==='a') return v==='High'?'badge bh':'badge bl';
  if(f==='p'){{if(v==='Casual')return'badge bc';if(v==='Competitive')return'badge bco';return'badge bhc';}}
  if(f==='c'){{if(v==='Low Engagement')return'badge ble';if(v==='Balanced')return'badge bba';return'badge bhe';}}
}}
function filt(){{
  const fa=document.getElementById('fA').value,fr=document.getElementById('fR').value;
  const fp=document.getElementById('fP').value,fc=document.getElementById('fC').value;
  let rows=all;
  if(fa) rows=rows.filter(r=>r.addiction_lr===fa);
  if(fr) rows=rows.filter(r=>r.addiction_rf===fr);
  if(fp) rows=rows.filter(r=>r.player_type===fp);
  if(fc) rows=rows.filter(r=>r.cluster===fc);
  render(rows);
}}
function render(rows){{
  if(!rows.length){{document.getElementById('tbody').innerHTML='<tr><td colspan="15" class="empty">No records found.</td></tr>';return;}}
  document.getElementById('tbody').innerHTML=rows.map(r=>{{
    const agree=r.addiction_lr===r.addiction_rf;
    return`<tr>
      <td style="color:#d1d5db">${{r.id}}</td>
      <td><span class="av">${{r.name[0].toUpperCase()}}</span>${{r.name}}</td>
      <td>${{r.age}}</td>
      <td><span style="font-size:11px;padding:2px 8px;border-radius:20px;background:${{r.player_category==='Professional'?'#f0fdf4':'#eff6ff'}};color:${{r.player_category==='Professional'?'#16a34a':'#1d4ed8'}};font-weight:600">${{r.player_category}}</span></td>
      <td>${{r.game_type}}</td>
      <td><strong>${{r.playtime}}h</strong></td><td>${{r.sleep}}h</td><td>${{r.rage_level}}</td><td>${{r.productivity_drop}}%</td>
      <td><span class="${{badge('a',r.addiction_lr)}}">${{r.addiction_lr}}</span></td>
      <td><span class="${{badge('a',r.addiction_rf)}}">${{r.addiction_rf}}</span></td>
      <td style="text-align:center">${{agree?'✅':'<span class="diff-badge">DIFF</span>'}}</td>
      <td><span class="${{badge('p',r.player_type)}}">${{r.player_type}}</span></td>
      <td><span class="${{badge('c',r.cluster)}}">${{r.cluster}}</span></td>
      <td style="color:#d1d5db;font-size:11px">${{r.created?r.created.slice(0,16):'—'}}</td>
    </tr>`;
  }}).join('');
}}
async function load(){{
  try{{
    const res=await fetch('/records');all=await res.json();
    const t=all.length,h=all.filter(r=>r.addiction_lr==='High').length;
    const diff=all.filter(r=>r.addiction_lr!==r.addiction_rf).length;
    const ap=t?(all.reduce((s,r)=>s+r.playtime,0)/t).toFixed(1):'—';
    const as_=t?(all.reduce((s,r)=>s+r.sleep,0)/t).toFixed(1):'—';
    document.getElementById('s-total').textContent=t;
    document.getElementById('s-high').textContent=h;
    document.getElementById('s-diff').textContent=diff;
    document.getElementById('s-play').textContent=t?ap+'h':'—';
    document.getElementById('s-sleep').textContent=t?as_+'h':'—';
    filt();
  }}catch(e){{document.getElementById('tbody').innerHTML='<tr><td colspan="15" class="empty">Cannot connect to Flask.</td></tr>';}}
}}
load();
</script></body></html>"""

# =============================================================================
# ML Models page
# =============================================================================
@app.route("/models")
def page_models():
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GamerIQ — ML Models</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>{SHARED_CSS}{NAV_CSS}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;}}
.sec{{margin-bottom:2.5rem;}}
.sh{{display:flex;align-items:center;gap:12px;margin-bottom:1rem;flex-wrap:wrap;}}
.si{{font-size:24px;}}.st{{font-size:18px;font-weight:700;font-family:'Orbitron',sans-serif;color:#1e1b4b;}}
.sa{{font-size:12px;padding:4px 12px;border-radius:20px;font-weight:700;}}
.alr{{background:#ede9fe;color:#5b21b6;}}.arf{{background:#fce7f3;color:#be185d;}}
.adt{{background:#dcfce7;color:#15803d;}}.akm{{background:#fef3c7;color:#b45309;}}
canvas{{max-height:260px;}}
.sr{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f3f4f6;font-size:13px;}}
.sr:last-child{{border-bottom:none;}}.sk{{color:#9ca3af;font-weight:500;}}.sv2{{color:#1e1b4b;font-weight:600;}}
.exp{{background:rgba(255,255,255,0.8);border:1.5px solid rgba(99,102,241,0.15);border-radius:12px;padding:1.25rem;font-size:13px;color:#6b7280;line-height:1.7;margin-top:1rem;}}
.exp strong{{color:#1e1b4b;}}
.leg{{display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;}}
.li{{display:flex;align-items:center;gap:6px;font-size:12px;color:#6b7280;font-weight:500;}}
.ld{{width:12px;height:12px;border-radius:50%;flex-shrink:0;}}
.compare-section{{background:white;border:1.5px solid rgba(99,102,241,0.2);border-radius:16px;padding:1.5rem;margin-bottom:2rem;box-shadow:0 4px 20px rgba(99,102,241,0.08);}}
.user-table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:1rem;}}
.user-table th{{text-align:left;font-size:11px;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:.06em;padding:10px 12px;border-bottom:2px solid #e0e7ff;background:#f8f7ff;}}
.user-table td{{padding:9px 12px;border-bottom:1px solid #f3f4f6;color:#374151;}}
.user-table tr:last-child td{{border-bottom:none;}}
.user-table tr:hover td{{background:#fafaff;}}
.diff-row td{{background:#fff7ed!important;}}
</style></head><body>
{nav('models')}
<div class="con">
  <p class="pt">ML MODELS</p>
  <p class="ps">Model performance, feature analysis, and real user prediction comparison</p>

  <!-- USER COMPARISON SECTION -->
  <div class="compare-section">
    <p class="ct" style="font-size:13px;">⚔️ LR vs Random Forest — User Prediction Comparison</p>
    <p style="font-size:13px;color:#6b7280;margin-bottom:1rem;">Shows where the two models agreed or disagreed on real user submissions</p>
    <div class="g2" style="margin-bottom:1.5rem;">
      <div class="card" style="padding:1rem;">
        <p class="ct">Addiction predictions — all users</p>
        <canvas id="userCompareChart" style="max-height:220px;"></canvas>
      </div>
      <div class="card" style="padding:1rem;">
        <p class="ct">Agreement rate</p>
        <canvas id="agreePieChart" style="max-height:220px;"></canvas>
        <div style="margin-top:1rem;">
          <div class="sr"><span class="sk">Total predictions</span><span class="sv2" id="uc-total">—</span></div>
          <div class="sr"><span class="sk">Models agree</span><span class="sv2" id="uc-agree" style="color:#16a34a;">—</span></div>
          <div class="sr"><span class="sk">Models disagree</span><span class="sv2" id="uc-diff" style="color:#c2410c;">—</span></div>
        </div>
      </div>
    </div>
    <p class="ct">Per-user comparison table</p>
    <div style="overflow:auto;">
      <table class="user-table" id="userTable">
        <thead><tr>
          <th>#</th><th>Name</th><th>Playtime</th><th>Sleep</th><th>Rage</th>
          <th>LR Prediction</th><th>RF Prediction</th><th>Match?</th>
        </tr></thead>
        <tbody id="userTbody"><tr><td colspan="8" style="text-align:center;padding:2rem;color:#9ca3af;">Loading user data...</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- LR vs RF accuracy comparison -->
  <div class="sec">
    <div class="sh"><span class="si">🧠</span><span class="st">LR vs Random Forest — Accuracy</span>
      <span class="sa alr" id="lr-badge">LR: Loading...</span>
      <span class="sa arf" id="rf-badge">RF: Loading...</span>
    </div>
    <div class="g2">
      <div class="card"><p class="ct">LR Feature Coefficients</p><canvas id="lrCoef"></canvas></div>
      <div class="card"><p class="ct">RF Feature Importance</p><canvas id="rfImport"></canvas>
        <div style="margin-top:1rem;">
          <div class="sr"><span class="sk">LR Accuracy</span><span class="sv2" id="lr-acc">—</span></div>
          <div class="sr"><span class="sk">RF Accuracy</span><span class="sv2" id="rf-acc">—</span></div>
          <div class="sr"><span class="sk">High addiction in data</span><span class="sv2" id="lr-high">—</span></div>
          <div class="sr"><span class="sk">Training samples</span><span class="sv2" id="lr-train">—</span></div>
        </div>
      </div>
    </div>
    <div class="exp"><strong>LR vs RF:</strong> Logistic Regression is fast and interpretable — the coefficient chart shows exactly which features drive predictions. Random Forest combines 40 trees for more robust results. When they disagree, it usually means the case is borderline — near the decision boundary.</div>
  </div>

  <!-- Decision Tree -->
  <div class="sec">
    <div class="sh"><span class="si">🎯</span><span class="st">Decision Tree — Player Type</span><span class="sa adt" id="dt-badge">Loading...</span></div>
    <div class="g2">
      <div class="card"><p class="ct">Feature Importance</p><canvas id="dtImport"></canvas></div>
      <div class="card"><p class="ct">Class Distribution</p><canvas id="dtPie"></canvas>
        <div style="margin-top:1rem;">
          <div class="sr"><span class="sk">Accuracy</span><span class="sv2" id="dt-acc">—</span></div>
          <div class="sr"><span class="sk">Max depth</span><span class="sv2" id="dt-depth">—</span></div>
          <div class="sr"><span class="sk">Number of leaves</span><span class="sv2" id="dt-leaves">—</span></div>
        </div>
      </div>
    </div>
    <div class="exp"><strong>Decision Tree:</strong> Splits on the most informative feature at each node. Playtime dominates because player type is directly defined by hours/day. The fitted tree needs only {DT_INFO["max_depth"]} levels because player type is a direct playtime rule.</div>
  </div>

  <!-- K-Means -->
  <div class="sec">
    <div class="sh"><span class="si">📊</span><span class="st">Engagement Clusters</span><span class="sa akm">3 clusters · playtime rule</span></div>
    <div class="leg">
      <div class="li"><div class="ld" style="background:#6366f1"></div>Low Engagement (&lt;4h)</div>
      <div class="li"><div class="ld" style="background:#16a34a"></div>Balanced (4–8h)</div>
      <div class="li"><div class="ld" style="background:#ec4899"></div>High Engagement (&gt;8h)</div>
    </div>
    <div class="g2">
      <div class="card"><p class="ct">Playtime vs Sleep (training data)</p><canvas id="km1"></canvas></div>
      <div class="card"><p class="ct">Playtime vs Productivity Drop</p><canvas id="km2"></canvas></div>
    </div>
    <div class="exp"><strong>Clusters:</strong> Each dot = one synthetic gamer. Colour = cluster. Clusters are assigned by fixed playtime thresholds (under 4h, 4 to 8h, over 8h), so they form vertical bands. No K-Means model is fitted.</div>
  </div>
</div>

<script>
async function load(){{
  let mi, records;
  try{{
    const [r1,r2]=await Promise.all([fetch('/model-info'),fetch('/records')]);
    mi=await r1.json(); records=await r2.json();
  }}catch(e){{
    document.querySelector('.con').innerHTML+='<p style="color:#ef4444;margin-top:2rem;font-weight:600;">Cannot connect to Flask. Make sure python app.py is running.</p>';
    return;
  }}
  const lr=mi.lr,rf=mi.rf,dt=mi.dt,km=mi.km;

  // Badges
  document.getElementById('lr-badge').textContent='LR: '+lr.accuracy+'%';
  document.getElementById('rf-badge').textContent='RF: '+rf.accuracy+'%';
  document.getElementById('dt-badge').textContent=dt.accuracy+'%';
  document.getElementById('lr-acc').textContent=lr.accuracy+'%';
  document.getElementById('rf-acc').textContent=rf.accuracy+'%';
  document.getElementById('lr-high').textContent=lr.high_count+' / '+(lr.high_count+lr.low_count);
  document.getElementById('lr-train').textContent=lr.train_size;

  // ── User comparison ──
  const total=records.length;
  const agree=records.filter(r=>r.addiction_lr===r.addiction_rf).length;
  const diff=total-agree;
  document.getElementById('uc-total').textContent=total;
  document.getElementById('uc-agree').textContent=agree;
  document.getElementById('uc-diff').textContent=diff;

  if(total>0){{
    // Bar chart: LR High, LR Low, RF High, RF Low per user (last 10)
    const recent=records.slice(0,10).reverse();
    const labels=recent.map(r=>r.name.split(' ')[0]);
    new Chart(document.getElementById('userCompareChart'),{{
      type:'bar',
      data:{{labels,datasets:[
        {{label:'LR — High',data:recent.map(r=>r.addiction_lr==='High'?1:0),backgroundColor:'#ef444488',borderColor:'#ef4444',borderWidth:1}},
        {{label:'LR — Low', data:recent.map(r=>r.addiction_lr==='Low'?1:0), backgroundColor:'#16a34a88',borderColor:'#16a34a',borderWidth:1}},
        {{label:'RF — High',data:recent.map(r=>r.addiction_rf==='High'?1:0),backgroundColor:'#ec489988',borderColor:'#ec4899',borderWidth:1}},
        {{label:'RF — Low', data:recent.map(r=>r.addiction_rf==='Low'?1:0), backgroundColor:'#6366f188',borderColor:'#6366f1',borderWidth:1}},
      ]}},
      options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#374151',font:{{size:11}}}}}}}},
        scales:{{x:{{ticks:{{color:'#9ca3af'}},grid:{{color:'#f3f4f6'}}}},y:{{ticks:{{color:'#9ca3af',stepSize:1}},grid:{{color:'#f3f4f6'}},max:1.2}}}}}}
    }});

    new Chart(document.getElementById('agreePieChart'),{{
      type:'doughnut',
      data:{{labels:['Both agree','Disagree'],datasets:[{{data:[agree,diff],backgroundColor:['#16a34a88','#f59e0b88'],borderColor:['#16a34a','#f59e0b'],borderWidth:2}}]}},
      options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#374151'}}}}}}}}
    }});

    document.getElementById('userTbody').innerHTML=records.map(r=>{{
      const match=r.addiction_lr===r.addiction_rf;
      return`<tr class="${{match?'':'diff-row'}}">
        <td style="color:#d1d5db">${{r.id}}</td>
        <td style="font-weight:600">${{r.name}}</td>
        <td>${{r.playtime}}h</td><td>${{r.sleep}}h</td><td>${{r.rage_level}}</td>
        <td><span class="badge ${{r.addiction_lr==='High'?'bh':'bl'}}">${{r.addiction_lr}}</span></td>
        <td><span class="badge ${{r.addiction_rf==='High'?'bh':'bl'}}">${{r.addiction_rf}}</span></td>
        <td style="text-align:center;font-size:16px">${{match?'✅':'⚠️'}}</td>
      </tr>`;
    }}).join('');
  }} else {{
    document.getElementById('userCompareChart').parentElement.innerHTML='<p style="text-align:center;color:#9ca3af;padding:2rem;font-size:13px;">No user predictions yet. Submit data on the Predict page first.</p>';
    document.getElementById('agreePieChart').parentElement.innerHTML='';
    document.getElementById('userTbody').innerHTML='<tr><td colspan="8" style="text-align:center;padding:2rem;color:#9ca3af;">No user data yet.</td></tr>';
  }}

  // LR Coefficients
  const co=Object.keys(lr.coefficients),cv=Object.values(lr.coefficients);
  new Chart(document.getElementById('lrCoef'),{{type:'bar',data:{{labels:co.map(k=>k.replace(/_/g,' ')),datasets:[{{data:cv,backgroundColor:cv.map(v=>v>=0?'#6366f188':'#ef444488'),borderColor:cv.map(v=>v>=0?'#6366f1':'#ef4444'),borderWidth:1.5}}]}},
    options:{{responsive:true,indexAxis:'y',plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#9ca3af'}},grid:{{color:'#f3f4f6'}}}},y:{{ticks:{{color:'#374151'}},grid:{{color:'#f3f4f6'}}}}}}}}}}
  );

  // RF Feature Importance
  const ri=Object.keys(rf.feature_importance),rv=Object.values(rf.feature_importance).map(v=>Math.round(v*1000)/10);
  new Chart(document.getElementById('rfImport'),{{type:'bar',data:{{labels:ri.map(k=>k.replace(/_/g,' ')),datasets:[{{data:rv,backgroundColor:'#ec489988',borderColor:'#ec4899',borderWidth:1.5}}]}},
    options:{{responsive:true,indexAxis:'y',plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#9ca3af'}},grid:{{color:'#f3f4f6'}}}},y:{{ticks:{{color:'#374151'}},grid:{{color:'#f3f4f6'}}}}}}}}}}
  );

  // DT
  document.getElementById('dt-acc').textContent=dt.accuracy+'%';
  document.getElementById('dt-depth').textContent=dt.max_depth;
  document.getElementById('dt-leaves').textContent=dt.n_leaves;
  const fi=Object.keys(dt.feature_importance),fv=Object.values(dt.feature_importance).map(v=>Math.round(v*1000)/10);
  new Chart(document.getElementById('dtImport'),{{type:'bar',data:{{labels:fi.map(k=>k.replace(/_/g,' ')),datasets:[{{data:fv,backgroundColor:'#16a34a88',borderColor:'#16a34a',borderWidth:1.5}}]}},
    options:{{responsive:true,indexAxis:'y',plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#9ca3af'}},grid:{{color:'#f3f4f6'}}}},y:{{ticks:{{color:'#374151'}},grid:{{color:'#f3f4f6'}}}}}}}}}}
  );
  const dc=Object.keys(dt.class_counts),dv=Object.values(dt.class_counts);
  new Chart(document.getElementById('dtPie'),{{type:'doughnut',data:{{labels:dc,datasets:[{{data:dv,backgroundColor:['#6366f188','#16a34a88','#ec489988'],borderColor:['#6366f1','#16a34a','#ec4899'],borderWidth:2}}]}},
    options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#374151'}}}}}}}}
  }});

  // K-Means
  function cc(c){{if(c==='Low Engagement')return'#6366f1';if(c==='Balanced')return'#16a34a';return'#ec4899';}}
  function makeDS(pts,xk,yk){{
    const g={{'Low Engagement':[],'Balanced':[],'High Engagement':[]}};
    pts.forEach(p=>g[p.cluster].push({{x:p[xk],y:p[yk]}}));
    return Object.entries(g).map(([name,data])=>
      ({{label:name,data,backgroundColor:cc(name)+'66',borderColor:cc(name),pointRadius:4,pointHoverRadius:6}}));
  }}
  const sc=km.scatter;
  new Chart(document.getElementById('km1'),{{type:'scatter',data:{{datasets:makeDS(sc,'playtime','sleep')}},
    options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#374151',boxWidth:12}}}}}},
      scales:{{x:{{title:{{display:true,text:'Playtime (hrs)',color:'#6b7280'}},ticks:{{color:'#9ca3af'}},grid:{{color:'#f3f4f6'}}}},
               y:{{title:{{display:true,text:'Sleep (hrs)',color:'#6b7280'}},ticks:{{color:'#9ca3af'}},grid:{{color:'#f3f4f6'}}}}}}}}}}
  );
  const sc2=sc.map(p=>Object.assign({{prod:Math.max(0,Math.round((p.playtime-3)*8))}},p));
  new Chart(document.getElementById('km2'),{{type:'scatter',data:{{datasets:makeDS(sc2,'playtime','prod')}},
    options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#374151',boxWidth:12}}}}}},
      scales:{{x:{{title:{{display:true,text:'Playtime (hrs)',color:'#6b7280'}},ticks:{{color:'#9ca3af'}},grid:{{color:'#f3f4f6'}}}},
               y:{{title:{{display:true,text:'Productivity Drop (%)',color:'#6b7280'}},ticks:{{color:'#9ca3af'}},grid:{{color:'#f3f4f6'}}}}}}}}}}
  );
}}
load();
</script></body></html>"""

if __name__ == "__main__":
    app.run(debug=True)
