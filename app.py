import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="Sentiment of Scripture",
    layout="wide"
)
st.markdown("""
<style>

/* Mechanix 4.0 Dark Theology Theme */

.stApp {
    background:
        radial-gradient(circle at top left, #3a2600 0%, transparent 25%),
        radial-gradient(circle at bottom right, #1f1f1f 0%, transparent 30%),
        linear-gradient(135deg, #050505 0%, #0d0d0d 50%, #000000 100%);
    color: #f5f1e8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080808, #14110a);
    border-right: 1px solid #8a6a24;
}

/* Main titles */
h1 {
    color: #d4af37 !important;
    text-shadow: 0 0 12px rgba(212, 175, 55, 0.45);
    font-weight: 800;
}

h2, h3 {
    color: #f1d27a !important;
}

/* Body text */
p, label, div, span {
    color: #f5f1e8;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #111111, #1b160b);
    border: 1px solid #d4af37;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 0 18px rgba(212, 175, 55, 0.18);
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #7a5c16, #d4af37);
    color: black;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

/* Inputs */
.stTextInput input {
    background-color: #141414;
    color: #f5f1e8;
    border: 1px solid #8a6a24;
    border-radius: 10px;
}

/* Data table */
[data-testid="stDataFrame"] {
    border: 1px solid #8a6a24;
    border-radius: 12px;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #111111;
    color: #d4af37 !important;
    border-radius: 10px;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label {
    color: #d4af37 !important;
}

/* Horizontal rule */
hr {
    border-color: #8a6a24;
}

</style>
""", unsafe_allow_html=True)

st.title("⚜️ Quantifying Faith Through Data")

st.write("Mechanix 4.0 inspired scripture intelligence — emotion, theme, and meaning mapped through data.")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():

    url = "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/csv/KJV.csv"

    df = pd.read_csv(url)

    # Rename columns
    df.columns = ['book', 'chapter', 'verse', 'text']

    # Sentiment
    df['sentiment'] = df['text'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    # Subjectivity
    df['subjectivity'] = df['text'].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity
    )

    return df

df = load_data()

# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ Filters")

book_options = sorted(df['book'].unique())

default_books = [book for book in ['Genesis', 'Psalms'] if book in book_options]

books = st.sidebar.multiselect(
    "Select Books",
    book_options,
    default=default_books
)

sentiment_range = st.sidebar.slider(
    "Sentiment Range",
    -1.0,
    1.0,
    (-0.5, 0.5)
)

# ---------------- FILTER DATA ----------------

filtered_df = df[
    (df['book'].isin(books)) &
    (df['sentiment'].between(sentiment_range[0], sentiment_range[1]))
]

# ---------------- METRICS ----------------

st.subheader("📊 Emotional Metrics")
st.info(f"""
### 📊 What This Data Means

This dashboard uses AI sentiment analysis to measure the emotional tone of scripture.

- **Average Sentiment** measures emotional polarity:
  - `-1` = highly negative (anger, fear, destruction)
  - `0` = neutral
  - `+1` = highly positive (love, peace, hope)

Current score: **{round(filtered_df['sentiment'].mean(), 3)}**

- **Total Verses** shows how many verses are currently being analyzed based on the selected books and filters.

Current verses analyzed: **{len(filtered_df)}**

- **Average Subjectivity** measures how emotionally opinionated or expressive the language is:
  - `0` = objective/factual language
  - `1` = emotional/opinion-heavy language

Current subjectivity: **{round(filtered_df['subjectivity'].mean(), 3)}**

### 📚 Sentiment by Book

The chart compares the average emotional tone of each selected book.

In the current analysis:
- **Psalms** appears more emotionally positive and expressive.
- **Genesis** trends more neutral due to historical and narrative language.

This analysis does NOT determine theology or truth.
It simply maps emotional language patterns using natural language processing (NLP).
""")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Sentiment",
        round(filtered_df['sentiment'].mean(), 3)
    )

with col2:
    st.metric(
        "Total Verses",
        len(filtered_df)
    )

with col3:
    st.metric(
        "Average Subjectivity",
        round(filtered_df['subjectivity'].mean(), 3)
    )

# ---------------- SENTIMENT BY BOOK ----------------

st.subheader("📚 Sentiment by Book")

book_sentiment = (
    filtered_df
    .groupby('book')['sentiment']
    .mean()
    .sort_values()
)

height = max(3, len(book_sentiment) * 0.5)

fig, ax = plt.subplots(figsize=(10, height))

book_sentiment.plot(
    kind='barh',
    ax=ax
)

ax.set_title("Average Sentiment by Book")
ax.set_xlabel("Sentiment Score")
ax.set_ylabel("Book")
plt.tight_layout()
st.pyplot(fig)
st.subheader("🔥 Emotion Heatmap")

emotion_words = {
    "Love": "love|mercy|grace|peace|forgive",
    "Fear": "fear|afraid|terror|dread",
    "Wrath": "wrath|anger|judgment|destroy",
    "Death": "death|die|dead|blood",
    "Hope": "hope|faith|promise|salvation"
}

heatmap_data = []

for book in filtered_df['book'].unique():
    book_data = filtered_df[filtered_df['book'] == book]

    row = {"Book": book}

    for emotion, pattern in emotion_words.items():
        row[emotion] = book_data['text'].str.contains(
            pattern,
            case=False,
            na=False
        ).sum()

    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data).set_index("Book")

fig3, ax3 = plt.subplots(figsize=(10, 5))

im = ax3.imshow(heatmap_df, aspect="auto")

ax3.set_xticks(range(len(heatmap_df.columns)))
ax3.set_xticklabels(heatmap_df.columns)

ax3.set_yticks(range(len(heatmap_df.index)))
ax3.set_yticklabels(heatmap_df.index)

ax3.set_title("Emotional Theme Intensity by Book")

for i in range(len(heatmap_df.index)):
    for j in range(len(heatmap_df.columns)):
        ax3.text(j, i, heatmap_df.iloc[i, j], ha="center", va="center")

fig3.colorbar(im)

st.pyplot(fig3)
st.subheader("🧠 AI Theology Insights")

avg_sentiment = filtered_df['sentiment'].mean()
avg_subjectivity = filtered_df['subjectivity'].mean()

if avg_sentiment > 0.1:
    mood = "mostly positive, leaning toward hope, praise, or comfort."
elif avg_sentiment < -0.1:
    mood = "emotionally heavy, leaning toward fear, judgment, or conflict."
else:
    mood = "mostly neutral, suggesting narrative, law, history, or balanced emotional language."

top_emotion = heatmap_df.sum().idxmax()

st.markdown(f"""
### Interpretation

The selected scripture section is **{mood}**

The strongest detected theme is **{top_emotion}**.

The average subjectivity score is **{round(avg_subjectivity, 3)}**, meaning the language is 
{"highly emotional and expressive" if avg_subjectivity > 0.4 else "more factual, historical, or restrained"}.

This dashboard does not judge faith. It reveals language patterns using data.
""")

# ---------------- LOVE VS FEAR ----------------

st.subheader("❤️ Love vs 😨 Fear")

love_words = "love|mercy|grace|peace|forgive"
fear_words = "fear|wrath|destroy|judgment|evil|death"

love_count = filtered_df['text'].str.contains(love_words, case=False, na=False).sum()
fear_count = filtered_df['text'].str.contains(fear_words, case=False, na=False).sum()
total_checked = love_count + fear_count

love_percent = round((love_count / total_checked) * 100, 1) if total_checked > 0 else 0
fear_percent = round((fear_count / total_checked) * 100, 1) if total_checked > 0 else 0

col1, col2, col3 = st.columns(3)

col1.metric("❤️ Love/Mercy Mentions", love_count)
col2.metric("😨 Fear/Wrath Mentions", fear_count)
col3.metric("⚖️ Dominant Theme", "Fear/Wrath" if fear_count > love_count else "Love/Mercy")

st.write(
    f"In the selected books, **Love/Mercy themes** make up **{love_percent}%** "
    f"of these emotional keywords, while **Fear/Wrath themes** make up **{fear_percent}%**."
)

theme_df = pd.DataFrame({
    'Theme': ['Love / Mercy / Grace / Peace', 'Fear / Wrath / Judgment / Death'],
    'Mentions': [love_count, fear_count]
})

fig2, ax2 = plt.subplots(figsize=(9, 5))

bars = ax2.bar(theme_df['Theme'], theme_df['Mentions'])

ax2.set_title("Theme Comparison: Comfort vs Warning")
ax2.set_ylabel("Number of Verses Mentioning Theme")
ax2.set_xlabel("Theme Category")

for bar in bars:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(height),
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )

st.pyplot(fig2)

with st.expander("What does this chart mean?"):
    st.write("""
    This chart counts verses that contain emotionally charged words.

    **Love/Mercy** includes words like love, mercy, grace, peace, and forgive.

    **Fear/Wrath** includes words like fear, wrath, destroy, judgment, evil, and death.

    This does not prove doctrine. It shows word-patterns in the selected scripture text.
    """)

# ---------------- SEARCH VERSES ----------------

st.subheader("🔍 Search Verses")

search_term = st.text_input(
    "Search for a word or phrase",
    "love"
)

search_results = filtered_df[
    filtered_df['text'].str.contains(
        search_term,
        case=False,
        na=False
    )
]

st.write(f"Found {len(search_results)} verses.")

st.dataframe(
    search_results[
        ['book', 'chapter', 'verse', 'text', 'sentiment']
    ].head(50),
    use_container_width=True
)