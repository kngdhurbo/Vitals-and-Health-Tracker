import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
from typing import Tuple, Union, Optional
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Constants ---
PAGE_TITLE = "Vital Analytics Dashboard"
PAGE_ICON = "📊"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "collapsed"
MIN_AGE = 18
MAX_AGE = 100
MIN_DAYS = 3
MAX_DAYS = 30
DEFAULT_DAYS_TO_TRACK = 7
DEFAULT_HEIGHT_M = 1.7
DEFAULT_BMI = 25.0
DEFAULT_WEIGHT = 70.0
DEFAULT_CHOL = 200
DEFAULT_GLUCOSE = 90
DEFAULT_TRESTBPS = 120
DEFAULT_DIASTOLIC = 80
DEFAULT_THALACH = 150
BMI_MIN = 15.0
BMI_MAX = 50.0

# --- Page Configuration ---

# --- Page Configuration ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# --- Global Styles (Python-only CSS injection) ---
def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        /* Page background and base typography */
        .stApp {
            background: radial-gradient(1200px 600px at 10% 10%, rgba(255, 255, 255, 0.04), transparent),
                        radial-gradient(1000px 500px at 90% 10%, rgba(255, 255, 255, 0.04), transparent),
                        linear-gradient(180deg, #0f1115 0%, #0b0d12 100%);
        }
        .hero-title {
            font-size: clamp(28px, 4vw, 44px);
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #fff 0%, #a0aec0 50%, #60a5fa 100%);
            -webkit-background-clip: text; background-clip: text; color: transparent;
            margin: 0 0 6px 0;
        }
        .hero-subtitle { color: #9aa4b2; font-size: 0.98rem; margin-bottom: 18px; }

        /* Glass card styling for bordered containers */
        [data-testid="stVerticalBlockBorderWrapper"]{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.06);
            backdrop-filter: blur(10px);
        }

        /* Inputs */
        .stTextInput > div > div > input,
        .stNumberInput input {
            border-radius: 12px !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            background: rgba(17, 24, 39, 0.55) !important;
            color: #e5e7eb !important;
        }
        .stTextInput > div > div > input:focus,
        .stNumberInput input:focus {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.35) !important;
            border-color: #60a5fa !important;
        }

        /* Primary button */
        button[kind="primary"], .stButton > button {
            border-radius: 12px;
            height: 48px;
            font-weight: 700;
            background: linear-gradient(90deg, #ef4444 0%, #fb7185 100%);
            border: 0;
            box-shadow: 0 8px 20px rgba(239, 68, 68, 0.35);
        }
        .stButton > button:hover { filter: brightness(1.05); }

        /* Helper text */
        .muted { color: #9aa4b2; font-size: 0.9rem; }

        /* Badges under hero */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(96,165,250,0.12);
            color: #bfdbfe;
            border: 1px solid rgba(96,165,250,0.3);
            font-size: 12px;
            letter-spacing: 0.2px;
        }

        /* Feature cards */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 12px;
            margin: 8px 0 18px 0;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
            padding: 12px 14px;
        }
        .feature-card .icon { font-size: 20px; }
        .feature-card .title { color: #e5e7eb; font-weight: 700; margin-top: 6px; font-size: 0.95rem; }
        .feature-card .desc { color: #9aa4b2; font-size: 0.88rem; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_global_styles()

# --- 1. Load the Trained Models & Scaler ---
@st.cache_data
def load_models() -> Tuple[Optional[RandomForestClassifier], Optional[RandomForestClassifier], Optional[StandardScaler]]:
    """
    Loads the trained models and scaler from disk.
    Returns:
        Tuple containing heart model, diabetes model, and diabetes scaler.
    """
    try:
        heart_model = joblib.load('heart_model.joblib')
        diabetes_model = joblib.load('diabetes_model.joblib')
        diabetes_scaler = joblib.load('diabetes_scaler.joblib')
        return heart_model, diabetes_model, diabetes_scaler
    except FileNotFoundError:
        st.error("Error: Model/Scaler files not found. Ensure they are in the app's root folder.")
        return None, None, None

heart_model, diabetes_model, diabetes_scaler = load_models()

# --- 2. The Prediction Function ---
def analyze_tracked_data(avg_vitals: pd.Series) -> list:
    """
    Analyze average vitals and return risk predictions for diseases.
    Args:
        avg_vitals (pd.Series): Average vitals for the tracked period.
    Returns:
        List of dictionaries with disease name, risk, and tip.
    """
    results = []
    # Hypertension risk assessment
    if avg_vitals.get('trestbps', 0) > 130 or avg_vitals.get('BloodPressure', 0) > 80:
        risk = min(max(((avg_vitals.get('trestbps', 130) - 130) / 70 * 100), ((avg_vitals.get('BloodPressure', 80) - 80) / 40 * 100)), 95)
        results.append({"name": "Hypertension", "risk": risk, "tip": "Your average blood pressure is high. Reduce sodium intake and consult a doctor."})
    else:
        results.append({"name": "Hypertension", "risk": np.random.uniform(5, 15), "tip": ""})
    # Obesity risk assessment
    if avg_vitals.get('BMI', 0) >= 30:
        risk = min(((avg_vitals.get('BMI', 30) - 30) / 20 * 100), 95)
        results.append({"name": "Obesity", "risk": risk, "tip": "Your average BMI is in the obese range. Focus on a balanced diet and regular exercise."})
    else:
        results.append({"name": "Obesity", "risk": np.random.uniform(5, 15), "tip": ""})

    # ML Model Predictions
    if heart_model is not None and diabetes_model is not None and diabetes_scaler is not None:
        vitals_dict = avg_vitals.to_dict()
        heart_features_df = pd.DataFrame([vitals_dict])
        cp_mode, thal_mode = int(vitals_dict.get('cp', 0)), int(vitals_dict.get('thal', 1))
        # One-hot encoding for categorical features
        for i in range(4):
            heart_features_df[f'cp_{i}'] = 1 if i == cp_mode else 0
        heart_features_df['thal_fixed defect'] = 1 if thal_mode == 2 else 0
        heart_features_df['thal_normal'] = 1 if thal_mode == 1 else 0
        heart_features_df['thal_reversable defect'] = 1 if thal_mode == 3 else 0
        heart_req_cols = heart_model.feature_names_in_
        for col in heart_req_cols:
            if col not in heart_features_df:
                heart_features_df[col] = 0
        heart_pred_proba = heart_model.predict_proba(heart_features_df[heart_req_cols])[0]
        results.append({"name": "Heart Disease", "risk": heart_pred_proba[1] * 100, "tip": "A diet low in saturated fats is crucial for heart health."})
        
        diabetes_features = pd.DataFrame([{
            "Glucose": avg_vitals.get('Glucose', DEFAULT_GLUCOSE),
            "BloodPressure": avg_vitals.get('BloodPressure', DEFAULT_DIASTOLIC),
            "BMI": avg_vitals.get('BMI', DEFAULT_BMI),
            "Age": avg_vitals.get('age', MIN_AGE)
        }])
        diabetes_features_scaled = diabetes_scaler.transform(diabetes_features)
        diabetes_pred_proba = diabetes_model.predict_proba(diabetes_features_scaled)[0]
        results.append({"name": "Diabetes", "risk": diabetes_pred_proba[1] * 100, "tip": "Monitor carbohydrate intake and consult your doctor."})

    return sorted(results, key=lambda x: x['risk'], reverse=True)

# --- 3. UI Helper Functions ---
def synced_input(label: str, min_val: float, max_val: float, default_val: float, step: float, key_suffix: str) -> float:
    """
    Create a synced slider and number input for a vital sign.
    Args:
        label (str): Label for the input.
        min_val (float): Minimum value.
        max_val (float): Maximum value.
        default_val (float): Default value.
        step (float): Step size.
        key_suffix (str): Suffix for session state keys.
    Returns:
        float: The value entered by the user.
    """
    slider_key = f"{key_suffix}_{st.session_state.current_day}_slider"
    number_key = f"{key_suffix}_{st.session_state.current_day}_number"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state.daily_logs.get(st.session_state.current_day, {}).get(key_suffix, default_val)
    if number_key not in st.session_state:
        st.session_state[number_key] = st.session_state.daily_logs.get(st.session_state.current_day, {}).get(key_suffix, default_val)
    def slider_callback():
        st.session_state[number_key] = st.session_state[slider_key]
    def number_callback():
        st.session_state[slider_key] = st.session_state[number_key]
    col1, col2 = st.columns([2, 1])
    with col1:
        st.slider(label, min_val, max_val, key=slider_key, on_change=slider_callback)
    with col2:
        st.number_input(label, min_val, max_val, key=number_key, on_change=number_callback, label_visibility="collapsed")
    return st.session_state[number_key]

# --- 4. Main App Logic ---

if 'screen' not in st.session_state:
    st.session_state.screen = 'welcome'
    st.session_state.daily_logs = {}
    st.session_state.current_day = 1
    st.session_state.days_to_track = 7
    st.session_state.profile = {
        'username': '', 'age': 45, 'sex': 0,
        'bmi_choice': "Calculate for me (requires daily weight)",
        'height_m': 1.7, 'manual_bmi': 25.0
    }

if st.session_state.screen == 'welcome':
    # Hero header
    st.markdown("<div class='hero-title'>Vital Analytics Dashboard: Personal Health Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Start a personalized, privacy-first session to track your daily vitals and get AI-backed insights.</div>", unsafe_allow_html=True)

    # Badges row
    st.markdown(
        """
        <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin: 4px 0 8px 0;">
            <span class="badge">🔒 Privacy-first</span>
            <span class="badge">⚡ Lightweight</span>
            <span class="badge">🤖 AI-backed</span>
            <span class="badge">📊 Actionable</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature highlights
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="icon">📈</div>
                <div class="title">Smart Trends</div>
                <div class="desc">Track key vitals over days to spot meaningful patterns.</div>
            </div>
            <div class="feature-card">
                <div class="icon">🧠</div>
                <div class="title">Risk Insights</div>
                <div class="desc">Get model-assisted estimates for common conditions.</div>
            </div>
            <div class="feature-card">
                <div class="icon">📝</div>
                <div class="title">Quick Logging</div>
                <div class="desc">Minimal inputs with synced sliders and number fields.</div>
            </div>
            <div class="feature-card">
                <div class="icon">⬇️</div>
                <div class="title">Shareable Report</div>
                <div class="desc">Export a clean PDF summary of your session.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("**Step 1: Start a New Tracking Session**")
            username = st.text_input("Enter your name (for saving your data):", st.session_state.profile.get('username', ''))
            days_to_track = st.number_input("How many days would you like to track?", min_value=MIN_DAYS, max_value=MAX_DAYS, value=st.session_state.days_to_track, step=1)
            st.markdown("<span class='muted'>Tip: Tracking for at least a week helps reveal meaningful trends.</span>", unsafe_allow_html=True)
        with right:
            st.markdown("#### Import an existing session")
            st.markdown("<span class='muted'>Upload an Excel/CSV with columns like: trestbps, BloodPressure, Glucose, BMI. Optional: age, sex, thalach, chol.</span>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Excel (.xlsx/.xls) or CSV", type=["xlsx", "xls", "csv"], label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                        df_import = pd.read_excel(uploaded_file)
                    else:
                        df_import = pd.read_csv(uploaded_file)

                    # Normalize column names for matching
                    original_columns = list(df_import.columns)
                    lower_to_original = {str(c).strip().lower(): c for c in original_columns}

                    def get_col(*candidates: str) -> Optional[str]:
                        for cand in candidates:
                            key = cand.strip().lower()
                            if key in lower_to_original:
                                return lower_to_original[key]
                        return None

                    # Resolve expected columns (with fallbacks)
                    col_trestbps = get_col("trestbps", "systolic", "systolic bp", "systolic_bp")
                    col_diastolic = get_col("bloodpressure", "diastolic", "diastolic bp", "diastolic_bp")
                    col_glucose = get_col("glucose", "glucose level", "glucose_level")
                    col_bmi = get_col("bmi")
                    col_thalach = get_col("thalach", "max heart rate", "max_hr", "heart_rate")
                    col_chol = get_col("chol", "cholesterol")
                    col_age = get_col("age")
                    col_sex = get_col("sex", "gender")

                    required_present = [c for c in [col_trestbps, col_diastolic, col_glucose, col_bmi] if c is not None]
                    if len(required_present) < 2:
                        st.warning("Please include at least two of these columns: trestbps, BloodPressure, Glucose, BMI.")
                    else:
                        # Build daily_logs mapping from rows
                        daily_logs = {}
                        num_rows = len(df_import)
                        profile_age = st.session_state.profile.get('age', 45)
                        profile_sex = st.session_state.profile.get('sex', 0)
                        default_bmi = st.session_state.profile.get('manual_bmi', DEFAULT_BMI)
                        for idx in range(num_rows):
                            row = df_import.iloc[idx]
                            glucose_val = float(row[col_glucose]) if col_glucose is not None and pd.notna(row[col_glucose]) else DEFAULT_GLUCOSE
                            bmi_val = float(row[col_bmi]) if col_bmi is not None and pd.notna(row[col_bmi]) else default_bmi
                            trestbps_val = float(row[col_trestbps]) if col_trestbps is not None and pd.notna(row[col_trestbps]) else DEFAULT_TRESTBPS
                            diastolic_val = float(row[col_diastolic]) if col_diastolic is not None and pd.notna(row[col_diastolic]) else DEFAULT_DIASTOLIC
                            thalach_val = float(row[col_thalach]) if col_thalach is not None and pd.notna(row[col_thalach]) else DEFAULT_THALACH
                            chol_val = float(row[col_chol]) if col_chol is not None and pd.notna(row[col_chol]) else DEFAULT_CHOL
                            age_val = int(row[col_age]) if col_age is not None and pd.notna(row[col_age]) else int(profile_age)
                            # Map sex: allow numeric 0/1 or string Male/Female
                            if col_sex is not None and pd.notna(row[col_sex]):
                                sex_raw = row[col_sex]
                                if isinstance(sex_raw, str):
                                    sex_val = 1 if sex_raw.strip().lower().startswith('m') else 0
                                else:
                                    sex_val = int(sex_raw)
                            else:
                                sex_val = int(profile_sex)

                            daily_logs[idx + 1] = {
                                'age': age_val,
                                'sex': sex_val,
                                'BMI': bmi_val,
                                'trestbps': trestbps_val,
                                'BloodPressure': diastolic_val,
                                'chol': chol_val,
                                'Glucose': glucose_val,
                                'thalach': thalach_val,
                                'fbs': 1 if glucose_val > 120 else 0,
                                'exang': 0,
                                'cp': 0,
                                'ca': 0,
                                'thal': 1,
                            }

                        # Update session and jump to report
                        st.session_state.daily_logs = daily_logs
                        st.session_state.days_to_track = len(daily_logs)
                        if not st.session_state.profile.get('username'):
                            inferred_name = (uploaded_file.name.split('.')[0])[:20]
                            st.session_state.profile['username'] = inferred_name or 'Imported'
                        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                        st.session_state.session_id = f"{st.session_state.profile['username']}_{timestamp}"
                        st.session_state.screen = 'report'
                        st.rerun()
                except Exception as e:
                    st.warning(f"Could not import file: {e}")

    if st.button("Begin Setup", type="primary", use_container_width=True):
        if username:
            st.session_state.profile['username'] = username
            st.session_state.days_to_track = days_to_track
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            st.session_state.session_id = f"{username}_{timestamp}"
            st.session_state.screen = 'profile_setup'
            st.rerun()
        else:
            st.warning("Please enter a name.")

elif st.session_state.screen == 'profile_setup':
    st.header("Step 2: Your Personal Profile")
    st.info("This information is collected once and used for all daily analyses.")
    
    bmi_choice = st.radio("How would you like to provide your BMI?", ("Calculate for me (requires daily weight)", "I'll enter my BMI manually"), key="bmi_choice_radio", index=["Calculate for me (requires daily weight)", "I'll enter my BMI manually"].index(st.session_state.profile['bmi_choice']))
    st.session_state.profile['bmi_choice'] = bmi_choice
    
    with st.form(key="profile_form"):
        age = st.number_input("Age (Years)", 18, 100, st.session_state.profile['age'])
        sex = st.radio("Sex", ('Male', 'Female'), index=st.session_state.profile['sex'])
        
        height = int(st.session_state.profile['height_m'] * 100)
        manual_bmi = st.session_state.profile['manual_bmi']
        
        if st.session_state.profile['bmi_choice'] == "Calculate for me (requires daily weight)":
            height = st.number_input("Your Height (cm)", 100, 250, height)
        else:
            manual_bmi = st.number_input("Your usual BMI", 15.0, 50.0, manual_bmi, 0.1)
        
        submitted = st.form_submit_button("Save Profile & Start Tracking", use_container_width=True, type="primary")
        if submitted:
            st.session_state.profile['age'] = age
            st.session_state.profile['sex'] = 1 if sex == 'Male' else 0
            if st.session_state.profile['bmi_choice'] == "Calculate for me (requires daily weight)":
                st.session_state.profile['height_m'] = height / 100
            else:
                st.session_state.profile['manual_bmi'] = manual_bmi
            st.session_state.screen = 'tracking'
            st.rerun()

elif st.session_state.screen == 'tracking':
    day = st.session_state.current_day
    st.header(f"Step 3: Log Vitals for Day {day} of {st.session_state.days_to_track}")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("General Vitals")
            if st.session_state.profile['bmi_choice'] == "Calculate for me (requires daily weight)":
                weight = synced_input("Weight (kg)", 40.0, 200.0, 70.0, 0.5, "weight")
                bmi_today = weight / (st.session_state.profile['height_m'] ** 2) if st.session_state.profile['height_m'] > 0 else 0
                st.metric("Your Calculated BMI today", f"{bmi_today:.1f}")
            else:
                bmi_today = st.session_state.profile['manual_bmi']
            chol = synced_input("Cholesterol (mg/dl)", 100, 400, 200, 1, "chol")
            glucose = synced_input("Glucose Level (mg/dL)", 60, 200, 90, 1, "glucose")
        with col2:
            st.subheader("Cardiovascular")
            trestbps = synced_input("Systolic BP (mmHg)", 80, 220, 120, 1, "trestbps")
            blood_pressure_diastolic = synced_input("Diastolic BP (mmHg)", 40, 120, 80, 1, "diastolic")
            thalach = synced_input("Max Heart Rate", 60, 220, 150, 1, "thalach")

    if st.button(f"Log Data for Day {day}", use_container_width=True, type="primary"):
        st.session_state.daily_logs[day] = {
            'age': st.session_state.profile['age'], 'sex': st.session_state.profile['sex'],
            'BMI': bmi_today, 'trestbps': trestbps, 'BloodPressure': blood_pressure_diastolic,
            'chol': chol, 'Glucose': glucose, 'thalach': thalach, 'fbs': 1 if glucose > 120 else 0,
            'exang': 0, 'cp': 0, 'ca': 0, 'thal': 1
        }
        st.success(f"Data for Day {day} saved successfully!")
        if day < st.session_state.days_to_track:
            st.session_state.current_day += 1
            st.rerun()

    nav_cols = st.columns(2)
    with nav_cols[0]:
        if day > 1:
            if st.button("⬅ Previous Day", use_container_width=True):
                st.session_state.current_day -= 1; st.rerun()
    with nav_cols[1]:
        if day < st.session_state.days_to_track and day in st.session_state.daily_logs:
            if st.button("Next Day ➡", use_container_width=True):
                st.session_state.current_day += 1; st.rerun()

    if len(st.session_state.daily_logs) == st.session_state.days_to_track:
        st.markdown("---")
        if st.button("🎉 All Data Logged! Analyze My Results", use_container_width=True, type="primary"):
            st.session_state.screen = 'report'
            st.rerun()

elif st.session_state.screen == 'report':
    st.header("Your Final Health Assessment Report")
    with st.spinner("Running AI analysis on your tracked data..."):
        user_df = pd.DataFrame.from_dict(st.session_state.daily_logs, orient='index')
        csv_filename = f"{st.session_state.session_id}_data.csv"
        user_df.to_csv(csv_filename, index_label="day")
        avg_vitals = user_df.mean()
        predictions = analyze_tracked_data(avg_vitals)
    st.success(f"Analysis complete! Your tracked data has been saved to {csv_filename}")
    
    st.subheader("Risk Overview")
    metric_cols = st.columns(4)
    for i, disease in enumerate(predictions):
        with metric_cols[i]:
            risk = disease['risk']
            risk_level = "HIGH" if risk > 60 else "MODERATE" if risk > 30 else "LOW"
            
            color = "#ff4b4b" if risk_level == "HIGH" else "#ffc44b" if risk_level == "MODERATE" else "#2ecc71"
            arrow = "↑" if risk_level == "HIGH" else "↓" if risk_level == "LOW" else ""

            st.markdown(f"""
            <div style="border: 1px solid {color}; border-radius: 7px; padding: 15px; text-align: center;">
                <h3 style="margin-bottom: 5px; font-size: 1.1rem; color: #555;">{disease['name']}</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0; color: {color};">{risk:.1f}%</p>
                <p style="font-weight: bold; margin: 0; color: {color};">{arrow} {risk_level} RISK</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Personalized Risk Profile")
    chart_data = pd.DataFrame(predictions)
    chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('risk:Q', title="Risk Percentage", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('name:N', title="", sort='-x'),
        color=alt.Color('risk:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True), legend=None),
        tooltip=[alt.Tooltip('name', title='Disease'), alt.Tooltip('risk', title='Risk', format='.1f')]
    ).properties(height=alt.Step(40))
    st.altair_chart(chart, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Daily Vital Trends")
    trend_data = user_df[['trestbps', 'BloodPressure', 'Glucose', 'BMI']].copy()
    trend_data.index.name = 'Day'
    trend_data.reset_index(inplace=True)
    trend_data_melted = trend_data.melt('Day', var_name='Vital', value_name='Value')
    line_chart = alt.Chart(trend_data_melted).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('Day:O', title='Tracking Day'),
        y=alt.Y('Value:Q', title='Measured Value'),
        color=alt.Color('Vital:N', title='Vital Sign'),
        tooltip=['Day', 'Vital', 'Value']
    ).interactive().properties(title="Your Vitals Over the Tracking Period")
    st.altair_chart(line_chart, use_container_width=True)
    
    st.subheader("Recommendations")
    for disease in predictions:
        if disease['risk'] > 30:
            with st.expander(f"See recommendations for {disease['name']}"):
                st.info(f"💡 {disease['tip']}", icon="ℹ")
    
    # --- PDF Download ---
    def build_pdf_report(username: str, session_id: str, days_to_track: int, avg_vitals_series: pd.Series, predictions_list: list, csv_path: str) -> bytes:
        """Generate a concise PDF report and return it as bytes."""
        # Lazy imports so the app still works without reportlab installed
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, title="Health-AI Report")
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="TitleBold", fontSize=18, leading=22, spaceAfter=10))
        styles.add(ParagraphStyle(name="Muted", textColor="#6b7280", fontSize=9))

        elements = []
        elements.append(Paragraph("Health-AI: Personal Health Report", styles["TitleBold"]))
        subtitle = f"User: {username or 'Anonymous'}  •  Session: {session_id}  •  Days tracked: {days_to_track}"
        elements.append(Paragraph(subtitle, styles["Muted"]))
        elements.append(Spacer(1, 12))

        # Risk table
        risk_rows = [["Condition", "Risk (%)", "Level"]]
        for row in predictions_list:
            risk = float(row["risk"]) if isinstance(row.get("risk"), (int, float, np.floating)) else 0.0
            level = "HIGH" if risk > 60 else "MODERATE" if risk > 30 else "LOW"
            risk_rows.append([row["name"], f"{risk:.1f}", level])
        risk_table = Table(risk_rows, hAlign="LEFT")
        risk_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#9ca3af")),
            ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f9fafb")),
        ]))
        elements.append(Paragraph("Risk Overview", styles["Heading3"]))
        elements.append(risk_table)
        elements.append(Spacer(1, 10))

        # Average vitals table (selected)
        vitals_keys = ["trestbps", "BloodPressure", "Glucose", "BMI", "age"]
        vitals_rows = [["Metric", "Average"]]
        for key in vitals_keys:
            if key in avg_vitals_series:
                val = float(avg_vitals_series.get(key, 0))
                vitals_rows.append([key, f"{val:.1f}"])
        vitals_table = Table(vitals_rows, hAlign="LEFT")
        vitals_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#9ca3af")),
            ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f3f4f6")),
        ]))
        elements.append(Paragraph("Average Vitals", styles["Heading3"]))
        elements.append(vitals_table)
        elements.append(Spacer(1, 16))

        # Recommendations (only for moderate/high risk)
        recs = [
            (row["name"], float(row["risk"]) if isinstance(row.get("risk"), (int, float, np.floating)) else 0.0, row.get("tip", ""))
            for row in predictions_list
        ]
        recs = [(n, r, t) for (n, r, t) in recs if r > 30 and t]
        if recs:
            elements.append(Paragraph("Recommendations", styles["Heading3"]))
            rec_rows = [["Condition", "Risk", "Recommendation"]]
            for name, risk, tip in recs:
                rec_rows.append([name, f"{risk:.1f}%", tip])
            rec_table = Table(rec_rows, hAlign="LEFT", colWidths=[120, 60, 320])
            rec_table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#9ca3af")),
                ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f9fafb")),
            ]))
            elements.append(rec_table)
            elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"Raw data CSV saved at: {csv_path}", styles["Muted"]))

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    try:
        pdf_data = build_pdf_report(
            username=st.session_state.profile.get('username', ''),
            session_id=st.session_state.get('session_id', 'session'),
            days_to_track=st.session_state.days_to_track,
            avg_vitals_series=avg_vitals,
            predictions_list=predictions,
            csv_path=csv_filename,
        )
        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_data,
            file_name=f"{st.session_state.get('session_id','report')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except ModuleNotFoundError:
        st.info("PDF export requires the 'reportlab' package. Install with: pip install reportlab")
    except Exception as e:
        st.warning(f"Could not generate PDF: {e}")

    if st.button("Start New Tracking Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()