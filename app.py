import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from PIL import Image
# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="Sentiment of Scripture",
    layout="wide"
)
st.markdown("""
<h2 style='
color:#ff3b3b;
font-size:22px;
font-weight:200;
text-shadow:0 0 15px rgba(255,0,0,.6);
margin-bottom:25px;
'>
&#8679; Choose your books
</h2>
""", unsafe_allow_html=True)
hero = Image.open("hero.png")
st.image("hero.png", use_container_width=True)

st.markdown("""
<style>

/* MAIN APP */
.stApp{
    background-color: black;
    color: white;
}

/* SIDEBAR */
section[data-testid="stSidebar"]{
    background-color: #050505;
}

/* HEADINGS */
h1, h2, h3{
    color: white !important;
}

/* TEXT */
p, div, span, label{
    color: white;
}

/* METRIC BOXES */
[data-testid="metric-container"]{
    background: #0a0a0a;
    border: 1px solid #d4af37;
    border-radius: 18px;
    padding: 15px;
}

/* CHART CONTAINERS */
.element-container{
    border-radius: 18px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* MULTISELECT BOX */
[data-baseweb="select"]{
    background-color:#0a0a0a !important;
    border:1px solid #d4af37 !important;
    border-radius:12px !important;
    color:white !important;
}

/* DROPDOWN MENU */
div[role="listbox"]{
    background:#050505 !important;
    border:1px solid #d4af37 !important;
}

/* OPTIONS */
div[role="option"]{
    background:#050505 !important;
    color:white !important;
}

/* HOVER */
div[role="option"]:hover{
    background:#1a1a1a !important;
}

/* SELECTED TAGS */
[data-baseweb="tag"]{
    background:#d4af37 !important;
    color:black !important;
    border-radius:20px !important;
}

/* INPUT TEXT */
input{
    color:white !important;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* DROPDOWN POPUP */
div[data-baseweb="popover"] > div {
    background-color: #050505 !important;
    color: white !important;
    border: 1px solid #d4af37 !important;
}

/* OPTIONS */
ul {
    background-color: #050505 !important;
}

li {
    background-color: #050505 !important;
    color: white !important;
}

/* HOVER */
li:hover {
    background-color: #1a1a1a !important;
    color: #d4af37 !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<h3 style='
color:#ff3b3b;
font-size:32px;
font-weight:200;
text-shadow:0 0 12px rgba(255,0,0,.5);
margin-top:-10px;
margin-bottom:50px;
'>
Mechanix 4.0 inspired scripture intelligence — emotion, theme, and meaning mapped through data.
</h3>
""", unsafe_allow_html=True)


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

st.sidebar.markdown("⬇️ Click the arrow to explore books")

books = st.sidebar.multiselect(
    "Select Books",
    df['book'].unique(),
    default=['Genesis','Psalms']
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
