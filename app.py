import json, pickle
import numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from gam_utils import term_curve, term_contributions          # needed so the pickle can be loaded

st.set_page_config(page_title='Endemic Bird Predictor - Sri Lanka', page_icon='🐦', layout='wide')


@st.cache_resource
def load():
    return pickle.load(open('model.pkl', 'rb')), json.load(open('feature_info.json'))


pipe, info = load()
prep, model = pipe.named_steps['prep'], pipe.named_steps['gam']
NUM_LOG, NUM_PLAIN, CAT = info['num_log'], info['num_plain'], info['cat']
st_ = info['stats']

st.title('🐦 Will this sighting be a Sri Lankan endemic bird?')
st.caption('Generalized Additive Model (LogisticGAM) trained on GBIF bird sightings 2014-2024 joined with satellite / climate covariates.')

# ------------------------------------------------------------------ inputs
st.sidebar.header('Sighting conditions')
row = {}
row['stateProvince'] = st.sidebar.selectbox('District', info['provinces'], index=0)
row['LandCover_Class'] = st.sidebar.selectbox('Land-cover class (code)', info['landcover'], index=0)
for c in NUM_LOG + NUM_PLAIN:
    s = st_[c]
    if c == 'month':
        row[c] = st.sidebar.slider('Month', 1, 12, 6)
    elif c == 'individualCount':
        row[c] = st.sidebar.number_input('Individuals counted', min_value=1, max_value=int(max(s['p99'], 2)), value=int(max(s['median'], 1)))
    else:
        lo, hi = float(s['p01']), float(s['p99'])
        row[c] = st.sidebar.slider(c, lo, hi, float(np.clip(s['median'], lo, hi)), step=(hi - lo) / 200 or 0.01)
X_raw = pd.DataFrame([row])[info['features']]

# ------------------------------------------------------------------ prediction
proba = float(pipe.predict_proba(X_raw)[0, 1])
left, right = st.columns([1, 2])
with left:
    st.subheader('Prediction')
    st.metric('Estimated probability of an endemic species', f'{proba:.1%}')
    st.progress(min(max(proba, 0.0), 1.0))
    if proba >= info['threshold']:
        st.success('Endemic-like habitat conditions (above threshold)')
    else:
        st.info('Typical of widespread species (below threshold)')
    st.caption(f'Endemics are a small minority of sightings, so even good habitat gives modest probabilities. The decision threshold used for the label is {info["threshold"]:.2f} (chosen on validation data).')

# ------------------------------------------------------------------ local explanation
# Raw GAM term values carry arbitrary constants (absorbed by the intercept), so contributions are shown
# RELATIVE TO A TYPICAL SIGHTING: median conditions, most frequent district and land cover (all stored in the pipeline).
@st.cache_resource
def typical_sighting():
    inv = {c: {v: k for k, v in prep.cat_maps_[c].items()} for c in CAT}
    t = {c: prep.medians_[c] for c in NUM_LOG + NUM_PLAIN}
    t.update({c: inv[c][prep.cat_mode_[c]] for c in CAT})
    Xt = pd.DataFrame([t])[info['features']]
    contrib_t, b = term_contributions(model, prep.transform(Xt).values[0])
    return t, contrib_t, b


typ, contrib_typ, intercept = typical_sighting()
contrib_x, _ = term_contributions(model, prep.transform(X_raw).values[0])
delta = contrib_x - contrib_typ                                   # effect of each input vs. the typical sighting
base_logit = intercept + contrib_typ.sum()
sigmoid = lambda z: 1 / (1 + np.exp(-z))
with right:
    st.subheader('Why? Each input compared with a typical sighting (log-odds)')
    order = np.argsort(np.abs(delta))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(model.feature_names)[order], delta[order], color=['#2a9d8f' if v > 0 else '#e76f51' for v in delta[order]])
    ax.axvline(0, color='k', lw=.8); ax.set_xlabel('pushes towards non-endemic  <-  0  ->  pushes towards endemic')
    st.pyplot(fig)
    st.caption(f"Typical sighting = median conditions in {typ['stateProvince']}, land cover {typ['LandCover_Class']}: baseline probability {sigmoid(base_logit):.1%} "
               f"({base_logit:+.2f} log-odds). Baseline + sum of bars = {base_logit + delta.sum():+.2f} log-odds = this sighting.")

# ------------------------------------------------------------------ global shape functions
st.subheader('What the model learned: effect curve of one feature')
feat = st.selectbox('Feature', [n for n in model.feature_names], index=model.feature_names.index('elevation_m') if 'elevation_m' in model.feature_names else 0)
i = model.feature_names.index(feat)
x, pdep, ci = term_curve(model, i)
fig2, ax2 = plt.subplots(figsize=(9, 3.6))
if feat in CAT:
    inv = {v: k for k, v in prep.cat_maps_[feat].items()}
    labels = [inv[int(v)] for v in x]
    o = np.argsort(pdep)
    ax2.bar(np.array(labels)[o], pdep[o], yerr=[(pdep - ci[:, 0])[o], (ci[:, 1] - pdep)[o]], color='#457b9d')
    ax2.tick_params(axis='x', rotation=75)
else:
    xs = np.expm1(x) if feat in NUM_LOG else x
    ax2.plot(xs, pdep, color='#1d3557'); ax2.fill_between(xs, ci[:, 0], ci[:, 1], alpha=.25, color='#457b9d')
    ax2.axvline(row[feat], color='#e63946', ls='--', label='your input'); ax2.legend()
    if feat in NUM_LOG: ax2.set_xscale('symlog', linthresh=1)
ax2.axhline(0, color='grey', lw=.8); ax2.set_ylabel('effect on log-odds (95% CI)'); ax2.set_title(feat)
st.pyplot(fig2)

with st.expander('About this model and its limits'):
    st.markdown(f"""
* **Model:** `{info['best_model']}` - one smooth curve per feature added together (log-odds scale), so every prediction can be decomposed exactly as shown above.
* **Test-set performance (location-disjoint split):** PR-AUC {info['metrics']['PR_AUC']:.3f}, ROC-AUC {info['metrics']['ROC_AUC']:.3f}, F1 {info['metrics']['F1']:.3f}.
* **Limits:** trained on citizen-science sightings (biased towards accessible places); probabilities are relative to that sample; shows association, not causation; the endemic label comes from a hand-compiled species list.
* **Ethics:** the app takes environmental conditions, not coordinates, to avoid exposing precise localities of rare species.
""")
