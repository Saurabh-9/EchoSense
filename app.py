from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import random

nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Prepare stop words
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
if 'no' in stop_words:
    stop_words.remove('no')


def get_emotion_feedback(emotion):
    """Provides dynamic feedback based on detected emotion"""
    positive_quotes = [
        "Your positive energy is truly inspiring!",
        "Keep shining bright and spreading positivity!",
        "What a wonderful mindset! Keep embracing joy.",
        "Beautiful thoughts lead to beautiful outcomes.",
        "Your optimism is your superpower—keep it up!"
    ]

    positive_advice = [
        "• Write down three things you're grateful for today.",
        "• Share your happiness with a friend or family member.",
        "• Use this positive energy to tackle something challenging.",
        "• Appreciate your recent achievements, no matter how small."
    ]

    negative_quotes = [
        "It's okay not to be okay. Tough times don't last forever.",
        "You're stronger than you think; one step at a time is enough.",
        "Every setback is a setup for a comeback—stay hopeful.",
        "Be gentle with yourself; growth often follows discomfort.",
        "Better days are coming; keep moving forward."
    ]

    negative_advice = [
        "• Take three deep breaths and practice mindfulness.",
        "• Talk to someone you trust about your feelings.",
        "• Do something kind for yourself today.",
        "• Try light exercise or a nature walk."
    ]

    feedback = {
        'positive': {
            'quote': random.choice(positive_quotes),
            'advice': random.sample(positive_advice, 2)
        },
        'negative': {
            'quote': random.choice(negative_quotes),
            'advice': random.sample(negative_advice, 2)
        }
    }

    return feedback.get(emotion.lower(), {
        'quote': "Stay focused and embrace the journey.",
        'advice': [
            "• Take one small step forward.",
            "• Practice self-care and gratitude."
        ]
    })


def compute_sentiment_percentage(dd):
    """Compute sentiment score based on combined weights"""
    pos_weight = 0.7 * dd['pos']
    neg_weight = -0.7 * dd['neg']
    compound_weight = 0.6 * dd['compound']
    
    sentiment_score = pos_weight + neg_weight + compound_weight
    return round((1 + sentiment_score) * 50, 2)


@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    # Handle empty form submissions
    text1 = request.form.get('text1', '').strip().lower()
    if not text1:
        return render_template('form.html', error="Please enter some text to analyze.")

    # Process text by removing stopwords
    processed_doc1 = ' '.join([word for word in text1.split() if word not in stop_words])

    # Analyze sentiment
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)

    # Compute sentiment percentage and classify emotion
    compound = compute_sentiment_percentage(dd)
    emotion = "negative" if dd['compound'] < 0.05 else "positive"
    feedback = get_emotion_feedback(emotion)

    return render_template(
        'form.html',
        final=compound,
        text1=text1,
        emotion=emotion,
        quote=feedback['quote'],
        advice=feedback['advice']
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
