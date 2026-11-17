"""Application de démonstration : analyse de sentiments (polarité + émotions).
Lancer :  streamlit run app.py
Placez ce dossier (app) dans livrable_phase1/ : l'application lit ../data/processed/train.csv et entraîne
les modèles au premier lancement (environ 30 secondes), puis les enregistre dans app/models/."""
import os, re
import numpy as np, pandas as pd, joblib, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = next((p for p in (os.path.join(HERE, "..", "data", "processed", "train.csv"), os.path.join(HERE, "train.csv")) if os.path.exists(p)), None)
MODELS = os.path.join(HERE, "models")
POL = {"happiness": "positive", "fun": "positive", "enthusiasm": "positive", "love": "positive", "relief": "positive",
       "sadness": "negative", "anger": "negative", "hate": "negative", "neutral": "neutral", "surprise": "neutral", "empty": "neutral"}

def clean(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t); t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"\d+", " ", t); t = re.sub(r"[^a-z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

NEG_FORMS = re.compile(r"\b(unhappy|unhappiness|unloved|discontent|discontented)\b")
NOT_POS = re.compile(r"(?:\bnot|n't|\bnever)\s+(?:really\s+|very\s+|so\s+|too\s+|at all\s+)?(?:happy|glad|joyful|excited|content|enthusiastic)\b")

@st.cache_resource(show_spinner="Chargement des modèles (premier lancement : environ 30 secondes)…")
def load_models():
    files = [os.path.join(MODELS, f) for f in ("tfidf.joblib", "svm_emotions.joblib", "svm_polarite.joblib")]
    try:
        return tuple(joblib.load(f) for f in files)
    except Exception:
        if TRAIN is None: raise FileNotFoundError("train.csv introuvable : placez le dossier app dans livrable_phase1/")
        tr = pd.read_csv(TRAIN).dropna(subset=["clean", "emotion"])
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True); X = tf.fit_transform(tr.clean)
        se = LinearSVC(C=0.5).fit(X, tr.emotion); sp_ = LinearSVC(C=0.5).fit(X, tr.emotion.map(POL))
        try:
            os.makedirs(MODELS, exist_ok=True)
            for obj, f in zip((tf, se, sp_), files): joblib.dump(obj, f)
        except Exception: pass
        return tf, se, sp_

def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

st.set_page_config(page_title="Analyse de sentiments", page_icon="💬", layout="centered")
st.title("Analyse de sentiments")
st.caption("Polarité (3 classes) et émotions (11 classes) · modèle TF-IDF + SVM linéaire · textes en anglais")
tfidf, svm_e, svm_p = load_models()

with st.sidebar:
    st.header("Options")
    use_rule = st.checkbox("Règle de négation (correctif ciblé)", value=True,
                           help="Répond sadness pour « unhappy », « not happy »… Décochez pour voir le modèle seul.")
    st.markdown("**Exemples**")
    ex = st.radio("Choisir un exemple", ["", "I am so happy to see you again today", "I hate waiting in this long queue",
                  "I feel really unhappy today", "He gave an average performance at the match"], label_visibility="collapsed")

texte = st.text_area("Texte à analyser (en anglais)", value=ex, height=110, placeholder="Écrivez une phrase en anglais…")
if st.button("Analyser", type="primary"):
    if len(clean(texte).split()) < 3:
        st.warning("Écrivez au moins 3 mots (en anglais).")
    else:
        low = texte.lower()
        if use_rule and (NEG_FORMS.search(low) or NOT_POS.search(low)):
            st.metric("Polarité", "negative"); st.metric("Émotion", "sadness")
            st.info("Règle de négation appliquée : ce résultat vient d'une règle, pas du modèle.")
        else:
            x = tfidf.transform([clean(texte)])
            sc = softmax(svm_e.decision_function(x)[0]); emo = pd.Series(sc, index=svm_e.classes_).sort_values(ascending=False)
            c1, c2 = st.columns(2); c1.metric("Polarité", svm_p.predict(x)[0]); c2.metric("Émotion", emo.index[0])
            st.bar_chart(emo.head(5), horizontal=True)
            st.caption("Les scores sont relatifs (calculés à partir des sorties du SVM) : ce ne sont pas de vraies probabilités.")

with st.expander("Limites"):
    st.write("Le modèle est entraîné sur des textes généraux en anglais, pas sur des commentaires de réseaux sociaux. "
             "Une partie des étiquettes du jeu semble venir de mots-clés cachés (par exemple « average » → anger, « unhappy » → happiness) : "
             "le modèle réagit à ces mots et ne comprend ni la négation ni le sarcasme.")
