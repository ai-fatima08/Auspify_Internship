# FlixAI: Netflix Content Recommendation System

Pick a Netflix title and get the 12 most similar titles. You can also add up to five liked titles to "My taste" and get recommendations that fit all of them. Built with Python, scikit-learn, Flask, HTML, CSS and JavaScript.

## Files
| File | Purpose |
|---|---|
| `data/Dataset.xlsx` | Netflix catalogue |
| `ml/recommender.py` | The machine learning part: cleaning, features, similarity, evaluation |
| `app.py` | Flask server: shows the website and answers its requests |
| `static/index.html` | The whole website in one file: HTML, CSS and JavaScript |
| `static/media/` | Optional hero video |
| `requirements.txt` | Python libraries |

## Run
```bash
pip install -r requirements.txt
python ml/recommender.py     # trains the model, creates ml/model.pkl and ml/metrics.json
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
