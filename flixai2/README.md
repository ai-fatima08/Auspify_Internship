# FlixAI: Netflix Machine Learning Tasks

Task 1: Content Recommendation System. Task 2: Content Type Prediction (Movie or TV Show).

Pick a Netflix title and get the 12 most similar titles. You can also add up to five liked titles to "My taste" and get recommendations that fit all of them. Built with Python, scikit-learn, Flask, HTML, CSS and JavaScript.

## Files
| File | Purpose |
|---|---|
| `data/Dataset.xlsx` | Netflix catalogue |
| `ml/recommender.py` | Task 1 ML: cleaning, features, similarity, evaluation |
| `ml/type_model.py` | Task 2 ML: feature selection, encoding, 4 classifiers, cross-validation, final models |
| `app.py` | Flask server: shows the website and answers its requests |
| `static/index.html` | The whole website in one file: HTML, CSS and JavaScript |
| `static/media/` | Optional hero video |
| `requirements.txt` | Python libraries |

## Run
```bash
pip install -r requirements.txt
python ml/recommender.py     # Task 1: creates ml/model.pkl and ml/metrics.json
python ml/type_model.py      # Task 2: creates ml/type_model.pkl and ml/type_metrics.json (takes a minute or two)
python app.py                # then open http://127.0.0.1:5000
```

## Optional extras
- **Real posters:** get a free API key from themoviedb.org, then run the server with it set.
  Windows: `set TMDB_API_KEY=your_key` then `python app.py`. Mac/Linux: `TMDB_API_KEY=your_key python app.py`.
  Without a key the site uses typographic posters. Follow TMDB's terms of use (attribution is shown in the footer).
- **Hero video:** put a video you have the rights to at `static/media/hero.mp4` (see `static/media/README.txt`).
  Without it the animated poster wall is shown.

## How the model works
1. Clean the data and split genres.
2. Convert genres, director, country, rating, type, decade and length into numeric vectors (CountVectorizer, TF-IDF for directors and countries).
3. Weight each group (genre 5.0, director 1.2, country 1.0, type 1.0, rating 0.5, era 0.4, length 0.4) and compute cosine similarity.
4. Return the top matches for the chosen title. For "My taste", the vectors of the chosen titles are averaged into one profile and compared against the catalogue.
5. Evaluate on 1,500 random titles: genre precision@10, genre overlap, type consistency, diversity, coverage, with a random baseline.

The dataset has no plot descriptions, so similarity uses metadata only. Genre scores are partly circular because genres are a model input.

## Task 2: how it works
1. **Select features:** genres, whether a director is credited, rating, first country, release year. Title, ID, director name and date added are left out (reasons are listed on the website).
2. **Encode:** one-hot encoding for rating and country, multi-hot for genres, scaling for numbers.
3. **Train:** Logistic Regression, Decision Tree, Random Forest and Gradient Boosting.
4. **Evaluate:** 5-fold cross-validation, then a final check on a 20% hold-out the models never saw (accuracy, precision, recall, F1, ROC AUC, confusion matrix, permutation importance).
5. **Compare:** four scenarios. Duration ("90 min" or "2 Seasons") gives the answer away, and Netflix uses different genre names for movies and shows. So the models are also tested without duration, with neutral genre names, and with genres removed completely (strict).

Result: the realistic model scores about 98.7% on the hold-out and the strict model about 94.8% (always guessing Movie scores 69.7%).
The strict model relies mostly on whether a director is credited. Treat the realistic score with care: it is high largely because of how Netflix names genres.
