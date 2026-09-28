from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)

from src.preprocessing import clean_text


MODEL_PATH = (
    ROOT /
    "models" /
    "sentiment_svm_model.pkl"
)

TFIDF_PATH = (
    ROOT /
    "models" /
    "tfidf_vectorizer.pkl"
)


st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(
    show_spinner="Loading trained model..."
)
def load_artifacts():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Sentiment model is missing."
        )

    if not TFIDF_PATH.exists():

        raise FileNotFoundError(
            "TF-IDF vectorizer is missing."
        )

    model = joblib.load(
        MODEL_PATH
    )

    tfidf = joblib.load(
        TFIDF_PATH
    )

    return model, tfidf


def predict_reviews(
    model,
    tfidf,
    reviews
):

    cleaned_reviews = [
        clean_text(review)
        for review in reviews
    ]

    vectors = tfidf.transform(
        cleaned_reviews
    )

    return model.predict(
        vectors
    )


def find_review_column(df):

    possible_columns = [

        "Clean_Review",
        "Review",
        "review",
        "Review_Text",
        "review_text",
        "Text",
        "text",
        "Comment",
        "comment"

    ]

    for column in possible_columns:

        if column in df.columns:

            return column

    return None


# -------------------------------
# UI DESIGN
# -------------------------------

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 18px;
        background:
        linear-gradient(
            135deg,
            #172554,
            #312e81
        );
        color: white;
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin-bottom: .4rem;
    }

    .hero p {
        opacity: .9;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="hero">

    <h1>
    💬 Sentiment Analyzer
    for Product Reviews
    </h1>

    <p>
    Analyze customer feedback using
    Machine Learning and Natural
    Language Processing.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


try:

    model, tfidf = load_artifacts()

except Exception as error:

    st.error(
        str(error)
    )

    st.stop()


# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:

    st.header(
        "⚙️ Analysis Mode"
    )

    mode = st.radio(
        "Choose an option",
        [
            "Single Review",
            "Bulk CSV Analysis"
        ]
    )

    st.divider()

    st.caption(
        "Machine Learning Model"
    )

    st.caption(
        "Linear SVM + TF-IDF"
    )

    st.caption(
        "Classes: Positive • Neutral • Negative"
    )


# ==================================================
# SINGLE REVIEW
# ==================================================

if mode == "Single Review":

    st.subheader(
        "🔎 Analyze a Product Review"
    )

    review = st.text_area(
        "Enter your review",
        placeholder=
        "Example: The product quality is excellent and delivery was very fast.",
        height=160
    )

    if st.button(
        "✨ Analyze Sentiment",
        type="primary",
        use_container_width=True
    ):

        if not review.strip():

            st.warning(
                "Please enter a review first."
            )

        else:

            prediction = predict_reviews(
                model,
                tfidf,
                [review]
            )[0]

            if prediction == "Positive":

                st.success(
                    f"🟢 Predicted Sentiment: {prediction}"
                )

            elif prediction == "Negative":

                st.error(
                    f"🔴 Predicted Sentiment: {prediction}"
                )

            else:

                st.info(
                    f"🟡 Predicted Sentiment: {prediction}"
                )


# ==================================================
# BULK CSV
# ==================================================

else:

    st.subheader(
        "📊 Bulk Review Analysis"
    )

    st.write(
        """
        Upload a CSV containing customer reviews.
        The system will automatically classify every
        review as Positive, Neutral, or Negative.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Reviews CSV",
        type=["csv"]
    )

    if uploaded_file:

        try:

            df = pd.read_csv(
                uploaded_file
            )

        except Exception as error:

            st.error(
                f"Unable to read CSV: {error}"
            )

            st.stop()


        review_column = find_review_column(
            df
        )


        if review_column is None:

            st.error(
                """
                No review column was detected.

                Use a column such as:
                Clean_Review, Review,
                review, Review_Text or Text.
                """
            )

            st.stop()


        working_df = df.copy()

        working_df[
            review_column
        ] = (
            working_df[
                review_column
            ]
            .fillna("")
            .astype(str)
        )


        working_df = working_df[
            working_df[
                review_column
            ]
            .str.strip()
            .ne("")
        ].copy()


        if working_df.empty:

            st.warning(
                "No usable reviews found."
            )

            st.stop()


        with st.spinner(
            "Analyzing reviews..."
        ):

            predictions = predict_reviews(
                model,
                tfidf,
                working_df[
                    review_column
                ].tolist()
            )


        working_df[
            "Predicted_Sentiment"
        ] = predictions


        total = len(
            working_df
        )

        counts = (
            working_df[
                "Predicted_Sentiment"
            ]
            .value_counts()
        )


        positive = int(
            counts.get(
                "Positive",
                0
            )
        )

        neutral = int(
            counts.get(
                "Neutral",
                0
            )
        )

        negative = int(
            counts.get(
                "Negative",
                0
            )
        )


        # ---------------------------
        # METRICS
        # ---------------------------

        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Total Reviews",
            total
        )


        c2.metric(
            "🟢 Positive",
            positive,
            f"{positive / total:.1%}"
        )


        c3.metric(
            "🟡 Neutral",
            neutral,
            f"{neutral / total:.1%}"
        )


        c4.metric(
            "🔴 Negative",
            negative,
            f"{negative / total:.1%}"
        )


        # ---------------------------
        # CHART
        # ---------------------------

        st.subheader(
            "📈 Sentiment Overview"
        )


        chart_df = pd.DataFrame(
            {
                "Reviews": [
                    positive,
                    neutral,
                    negative
                ]
            },
            index=[
                "Positive",
                "Neutral",
                "Negative"
            ]
        )


        st.bar_chart(
            chart_df
        )


        # ---------------------------
        # REVIEW EXPLORER
        # ---------------------------

        st.subheader(
            "🔍 Explore Customer Feedback"
        )


        selected_sentiments = st.multiselect(

            "Select sentiments",

            [
                "Positive",
                "Neutral",
                "Negative"
            ],

            default=[
                "Positive",
                "Neutral",
                "Negative"
            ]

        )


        filtered_df = working_df[
            working_df[
                "Predicted_Sentiment"
            ].isin(
                selected_sentiments
            )
        ].copy()


        search_text = st.text_input(

            "Search reviews",

            placeholder=
            "Try: battery, camera, delivery..."

        )


        if search_text.strip():

            filtered_df = filtered_df[
                filtered_df[
                    review_column
                ].str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]


        st.write(
            f"Showing **{len(filtered_df)}** reviews"
        )


        st.dataframe(

            filtered_df[
                [
                    review_column,
                    "Predicted_Sentiment"
                ]
            ],

            use_container_width=True,

            hide_index=True,

            height=420

        )


        # ---------------------------
        # DOWNLOAD
        # ---------------------------

        csv_data = (
            filtered_df
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )


        st.download_button(

            "⬇️ Download Analysis Results",

            data=csv_data,

            file_name=
            "sentiment_analysis_results.csv",

            mime="text/csv",

            use_container_width=True

        )


        st.info(
            """
            💡 Business insight:
            Negative reviews can be filtered and
            inspected individually to identify
            customer problems and improvement areas.
            """
        )
