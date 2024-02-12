import re
import streamlit as st
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from googlesearch import search


nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')


# Function to perform Google search
def perform_google_search(query, num_results=20, lang=None, advanced=True, sleep_interval=5):
    search_results = []

    # Build the search query with specified options
    search_query = query
    if lang:
        search_query += f" lang:{lang}"

    if advanced:
        search_results = search(query, num_results=num_results, lang=lang, advanced=True)
    else:
        search_results = search(query, num_results=num_results)

    return search_results

# Function to scrape blog content
def scrape_blog(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP status codes >= 400

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements (e.g., script, style)
        for script in soup(['script', 'style']):
            script.decompose()

        # Get text from all paragraphs
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div', 'li', 'a', 'strong', 'em', 'blockquote', 'pre', 'i', 'u', 'code', 'abbr', 'cite', 'address', 'time', 'b', 's', 'mark', 'small', 'sub', 'sup'])
        blog_text = ' '.join([paragraph.get_text() for paragraph in paragraphs])

        # Remove HTML tags
        cleaned_text = re.sub(r'<.*?>', ' ', blog_text)

        return cleaned_text.strip()

    except requests.exceptions.RequestException as req_err:
        st.error(f'Request error occurred: {req_err}')
        return None
    except Exception as e:
        st.error(f'Error occurred while scraping the blog at {url}: {e}')
        return None

# Function for sentiment analysis
def sentiment_analysis(sentences):
    sid = SentimentIntensityAnalyzer()
    scores = [sid.polarity_scores(sentence)['compound'] for sentence in sentences]
    positive_sentences = [sentence for i, sentence in enumerate(sentences) if scores[i] > 0]
    negative_sentences = [sentence for i, sentence in enumerate(sentences) if scores[i] < 0]
    return positive_sentences, negative_sentences

# Function to extract entities
def extract_entities(sentences):
    words = [word_tokenize(sentence) for sentence in sentences]
    entities = [' '.join(word.strip() for word in sublist if len(word) > 1 and word.isalnum() and not word.isdigit()) for sublist in words]
    entities = [entity for entity in entities if entity]  # Remove empty strings
    if not entities:
        # If no entities with at least two words, use the longest entities with one word
        entities = [' '.join(max(words, key=len))]

    top_entities = Counter(entities).most_common(10)
    return top_entities

# Function to generate word cloud
def generate_word_cloud(entities):
    # Remove stopwords and other irrelevant words
    stopwords = set(STOPWORDS)
    irrelevant_words = set(["said", "says", "will", "one", "also", "like", "new", "use", "get", "make", "may", "the", "you", "Please"])
    stopwords.update(irrelevant_words)

    if not entities:
        st.warning("No entities to generate word cloud.")
        return

    # Remove extra spaces from entities and convert to lowercase
    entities_cleaned = [(entity.strip().lower(), count) for entity, count in entities]

    wordcloud = WordCloud(width=1600, height=800, background_color='black', stopwords=stopwords).generate_from_frequencies(dict(entities_cleaned))
    st.image(wordcloud.to_array())

def main():
    st.title("Blog Analysis App")

    query_to_search = st.text_input("Enter a query:")
    if st.button("Search"):
        search_results = perform_google_search(query_to_search, num_results=10, lang="en", advanced=True, sleep_interval=5)

        all_blog_text = ""
        all_sentences = []
        all_positive_sentences = []
        all_negative_sentences = []
        all_entities = []

        for result in search_results:
            url = result.url
            blog_text = scrape_blog(url)

            if blog_text is not None and blog_text != "":
                sentences = sent_tokenize(blog_text)

                # Accumulate blog text
                all_blog_text += blog_text

                # Accumulate sentences
                all_sentences.extend(sentences)

                # Perform sentiment analysis
                positive_sentences, negative_sentences = sentiment_analysis(sentences)
                all_positive_sentences.extend(positive_sentences)
                all_negative_sentences.extend(negative_sentences)

                # Perform NLP tasks
                entities = extract_entities(sentences)
                all_entities.extend(entities)

        # Display insights for all blogs
        st.subheader("Top Most Positive Sentences:")
        st.write(all_positive_sentences[:10])

        st.subheader("Top Most Negative Sentences:")
        st.write(all_negative_sentences[:10])

        st.subheader("Top 10 Unique Entities:")
        st.write(all_entities)

        st.subheader("Word Cloud of Entities:")
        generate_word_cloud(all_entities)

if __name__ == "__main__":
    main()
