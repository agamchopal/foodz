import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#import preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
from nltk.sentiment import SentimentIntensityAnalyzer


st.sidebar.title("Food Sentiments")
uploaded_file = st.sidebar.file_uploader("Choose a file")
st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

selected = option_menu(
    menu_title=None,
    options=["Home", "Dashboard"],
    icons=["house", "bar-chart-line-fill"],
    default_index=0,
    orientation="horizontal",
)
import pandas as pd
def preprocess(data):
    df=pd.read_csv("Documents.csv")
    
    return df
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.dataframe(df)

button_clicked = st.button("Generate Word Cloud of Reviews")

# Check if the button is clicked
if button_clicked:
    # Check if 'Review' column exists in the DataFrame
    if 'Review' in df.columns:
        # Combine all reviews into a single string
        reviews_text = ' '.join(df['Review'])

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews_text)

        # Display the word cloud using Streamlit
        st.image(wordcloud.to_image(), caption='Word Cloud of Reviews')
    else:
        st.warning("The 'Review' column does not exist in the DataFrame.")
button_clicked = st.button(" Number of times each word used Reviews")
if button_clicked:
    # Check if 'Review' column exists in the DataFrame
    if 'Review' in df.columns:
        df=pd.read_csv("Documents.csv")
        stop_words = set(stopwords.words('english'))

        tokenized_reviews = []
        for review in df['Review']:
            word_counts = Counter(tokenized_reviews)
            words = word_tokenize(review)

            #words = words_tokenize(review)
    #print(words)
            words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
            tokenized_reviews.extend(words)
            most_common_words = word_counts.most_common()
         #word_counts = Counter(tokenized_reviews)

        # Convert the results to a DataFrame
        #most_common_words_df = pd.DataFrame(word_counts.most_common(), columns=['Word', 'Count'])

        # Display the DataFrame in Streamlit
        st.dataframe(most_common_words)
    else:
        st.warning("The 'Review' column does not exist in the DataFrame.")
button_clicked = st.button(" Scatter Plot ")
if button_clicked:
    # Check if 'Review' column exists in the DataFrame
    if 'Review' in df.columns:
        def get_sentiment_polarity(text):
         sentiment = SentimentIntensityAnalyzer()
         sentiment_score = sentiment.polarity_scores(text)['compound']
         return sentiment_score
        df['Sentiment'] = df['Review'].apply(get_sentiment_polarity)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        scatter = ax.scatter(df.index, df['Sentiment'], alpha=0.5)
        ax.set_title('Scatter Plot of Sentiment')
        ax.set_xlabel('X Variable')  # Replace with an appropriate label for your x-variable
        ax.set_ylabel('Sentiment')

# Display the plot in Streamlit
        st.pyplot(fig)

# Assuming df5['Sentiment'] contains numerical sentiment values
# You may need to replace 'x_variable' with the appropriate variable you want to compare sentiment with
        '''x_variable = df.index  # Example: Using the DataFrame index as the x-variable

        plt.scatter(x_variable, df['Sentiment'], alpha=0.5)
        plt.title('Scatter Plot of Sentiment')
        plt.xlabel('X Variable')  # Replace with an appropriate label for your x-variable
        plt.ylabel('Sentiment')
        plt.show()
        st.pyplot(fig)'''


# Print or analyze the results
        ##for word, count in most_common_words:
         #print(f"{word}: {count}")


'''st.button("Word_Cloud")
if selected== "Word_Cloud":
    df=pd.read_csv("Documents.csv")
    reviews_text = ' '.join(df['Review'])
    print(df["Review"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews_text)

# Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Reviews')
    plt.show()'''

'''if uploaded_file is not None:
    bytes_data=uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    st.text(data)
    st.dataframe(df)'''
