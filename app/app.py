from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import clean_text

MODEL_PATH = ROOT / 'models' / 'sentiment_svm_model.pkl'
TFIDF_PATH = ROOT / 'models' / 'tfidf_vectorizer.pkl'

st.set_page_config(page_title='Sentiment Analyzer', page_icon='💬', layout='wide')

@st.cache_resource(show_spinner='Loading trained model...')
def load_artifacts():
    if not MODEL_PATH.exists() or not TFIDF_PATH.exists():
        raise FileNotFoundError('Model files are missing. Run: python src\\train_model.py')
    return joblib.load(MODEL_PATH), joblib.load(TFIDF_PATH)

def predict_reviews(model, tfidf, reviews):
    cleaned = [clean_text(x) for x in reviews]
    return model.predict(tfidf.transform(cleaned))

def find_review_column(df):
    for col in ['Clean_Review','Review','review','Review_Text','review_text','Text','text','comment','Comment']:
        if col in df.columns: return col
    return None

st.markdown('''<style>
.block-container{padding-top:1.5rem}.hero{padding:1.5rem;border-radius:18px;background:linear-gradient(135deg,#172554,#312e81);color:white;margin-bottom:1rem}.hero h1{margin:0 0 .4rem}.hero p{margin:0;opacity:.9}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>💬 Sentiment Analyzer for Product Reviews</h1><p>Analyze one review or upload a CSV to discover Positive, Neutral and Negative customer feedback.</p></div>', unsafe_allow_html=True)

try:
    model, tfidf = load_artifacts()
except Exception as e:
    st.error(str(e)); st.stop()

with st.sidebar:
    st.header('⚙️ Analysis')
    mode = st.radio('Choose mode', ['Single Review','Bulk CSV Analysis'])
    st.divider(); st.caption('Model: Linear SVM + TF-IDF'); st.caption('Classes: Positive • Neutral • Negative')

if mode == 'Single Review':
    st.subheader('🔎 Analyze one review')
    review = st.text_area('Product review', placeholder='Example: The battery lasts all day and the camera quality is excellent.', height=150)
    if st.button('✨ Analyze Sentiment', type='primary', use_container_width=True):
        if not review.strip(): st.warning('Please enter a review first.')
        else:
            prediction = predict_reviews(model, tfidf, [review])[0]
            if prediction == 'Positive': st.success(f'🟢 Predicted Sentiment: **{prediction}**')
            elif prediction == 'Negative': st.error(f'🔴 Predicted Sentiment: **{prediction}**')
            else: st.info(f'🟡 Predicted Sentiment: **{prediction}**')
else:
    st.subheader('📊 Bulk Review Analysis')
    st.write('Upload a CSV containing a review/text column. Every review will be classified and can be filtered, searched and downloaded.')
    uploaded = st.file_uploader('Upload reviews CSV', type=['csv'])
    if uploaded:
        try: df = pd.read_csv(uploaded)
        except Exception as e: st.error(f'Could not read the CSV: {e}'); st.stop()
        review_col = find_review_column(df)
        if review_col is None:
            st.error('No review column found. Use Clean_Review, Review, review, Review_Text, review_text, Text or text.'); st.stop()
        working = df.copy(); working[review_col] = working[review_col].fillna('').astype(str); working = working[working[review_col].str.strip().ne('')].copy()
        if working.empty: st.warning('The uploaded file contains no usable reviews.'); st.stop()
        with st.spinner('Analyzing reviews...'):
            working['Predicted_Sentiment'] = predict_reviews(model, tfidf, working[review_col].tolist())
        total = len(working); counts = working['Predicted_Sentiment'].value_counts(); positive=int(counts.get('Positive',0)); neutral=int(counts.get('Neutral',0)); negative=int(counts.get('Negative',0))
        c1,c2,c3,c4=st.columns(4); c1.metric('Total Reviews',total); c2.metric('🟢 Positive',positive,f'{positive/total:.1%}'); c3.metric('🟡 Neutral',neutral,f'{neutral/total:.1%}'); c4.metric('🔴 Negative',negative,f'{negative/total:.1%}')
        st.subheader('Sentiment Overview')
        st.bar_chart(pd.DataFrame({'Count':[positive,neutral,negative]},index=['Positive','Neutral','Negative']))
        st.subheader('🔍 Explore Feedback')
        selected=st.multiselect('Show sentiments',['Positive','Neutral','Negative'],default=['Negative','Neutral','Positive'])
        filtered=working[working['Predicted_Sentiment'].isin(selected)].copy()
        search=st.text_input('Search inside reviews',placeholder='battery, camera, delivery...')
        if search.strip(): filtered=filtered[filtered[review_col].str.contains(search,case=False,na=False)]
        st.write(f'Showing **{len(filtered)}** reviews')
        st.dataframe(filtered[[review_col,'Predicted_Sentiment']],use_container_width=True,hide_index=True,height=420)
        st.download_button('⬇️ Download filtered results',filtered.to_csv(index=False).encode('utf-8'),'sentiment_analysis_results.csv','text/csv',use_container_width=True)
        st.caption('The current model is trained for the language/domain represented by the supplied dataset. New domains or languages should be evaluated before production use.')
