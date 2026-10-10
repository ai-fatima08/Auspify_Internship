# FlixAI: Netflix Machine Learning Tasks

Task 1: Content Recommendation System. Task 2: Content Type Prediction (Movie or TV Show). Task 3: Audience Rating Classification. Task 4: Content Segmentation (clustering). Task 5: Trend Forecasting.

Pick a Netflix title and get the 12 most similar titles. You can also add up to five liked titles to "My taste" and get recommendations that fit all of them. Built with Python, scikit-learn, Flask, HTML, CSS and JavaScript.

## Files
| File | Purpose |
|---|---|
| `data/Dataset.xlsx` | Netflix catalogue |
| `ml/recommender.py` | Task 1 ML: cleaning, features, similarity, evaluation |
| `ml/forecast_model.py` | Task 5 ML: time features, history analysis, 9 forecasting models, backtesting, 12-month forecast |
| `ml/cluster_model.py` | Task 4 ML: feature preparation, K-Means, algorithm comparison, cluster profiles, t-SNE map |
| `ml/rating_model.py` | Task 3 ML: rating categories, 4 classifiers, hyperparameter tuning, evaluation |
| `ml/type_model.py` | Task 2 ML: feature selection, encoding, 4 classifiers, cross-validation, final models |
| `app.py` | Flask server: shows the website and answers its requests |
| `static/index.html` | The whole website in one file: HTML, CSS and JavaScript |
| `static/media/` | Optional hero video |
| `requirements.txt` | Python libraries |

## Run
```bash
pip install -r requirements.txt
python ml/recommender.py     # Task 1: creates ml/model.pkl and ml/metrics.json
python ml/forecast_model.py  # Task 5: creates ml/forecast_metrics.json (about 30 seconds)
python ml/cluster_model.py   # Task 4: creates ml/cluster_metrics.json and ml/cluster_points.json (about 2 minutes)
python ml/rating_model.py    # Task 3: creates ml/rating_model.pkl and ml/rating_metrics.json (takes about 3 minutes)
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

## Task 3: how it works
1. **Analyze rating categories:** the 14 raw ratings are grouped into 4 audience categories: Kids (TV-Y, TV-Y7, TV-Y7-FV, TV-G, G), Family (TV-PG, PG), Teens (TV-14, PG-13), Adults (TV-MA, R, NC-17). Unrated titles (NR, UR) are removed.
2. **Prepare training data:** stratified 80/20 split. Features: genres, movie or show, minutes or seasons, country, release year, director listed, number of genres.
3. **Train:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, each with default settings.
4. **Optimize:** RandomizedSearchCV (8 settings, 4-fold cross-validation, scored on macro F1) for every model, plus a depth curve for the decision tree.
5. **Evaluate:** the model with the best cross-validated macro F1 is tested once on the 20% hold-out (accuracy, macro and weighted F1, per-category precision and recall, confusion matrix, permutation importance).

Result: Random Forest, about 62.8% accuracy and 0.615 macro F1 on unseen titles, against 46.0% for always guessing Adults.
Family and Teens are the hardest categories because their titles overlap with Adults and Kids in genre and length.

## Task 4: how it works
1. **Prepare features:** genres (multi-hot), movie or show, audience rating group, country (one-hot), release year, minutes, seasons and number of genres (standardised). Each block has a weight, then PCA keeps 85% of the information (69 columns become 21 components).
2. **Cluster:** K-Means for k = 2 to 12 with elbow and silhouette curves. k is chosen as the best silhouette between 4 and 10 where every group holds at least 3% of titles (k = 9). K-Means is compared with Agglomerative (Ward) and Gaussian Mixture using silhouette, Davies-Bouldin and Calinski-Harabasz, and its stability is checked over five random starts.
3. **Identify groups:** each cluster is profiled (size, movie share, audience, median year, distinctive genres, countries, typical titles) and named automatically from its own data.
4. **Visualize:** a 2D t-SNE map (display only) coloured by group, with a slider for k, a title finder, and a heatmap comparing groups.
5. **Interpret:** the website shows a card and a written summary for every group.

Silhouette scores are about 0.2, which is normal for catalogue data where titles overlap a lot. The groups are useful tendencies, not hard boundaries.

## Task 5: how it works
The target is the number of titles Netflix adds each month (from `date_added`), as one series for all titles, one for movies and one for TV shows. The series runs from January 2016 to August 2021. The unfinished September 2021 is left out of training.
1. **Time-based features:** month as sine and cosine, quarter, time index, lags 1, 2, 3 and 12, rolling averages of 3 and 12 months. Each feature uses only earlier months.
2. **History analysis:** titles added per year, seasonal index by month, autocorrelation, and the share of additions released within a year.
3. **Models (9):** Naive, Seasonal naive, Moving average; Linear trend with month effects and Holt-Winters (damped trend, written from scratch because statsmodels is not required); Ridge, Random Forest and Gradient Boosting on the lag features; and an Ensemble that averages five of them.
4. **Forecast:** the model with the lowest rolling-origin backtest error is refitted on all months and forecasts the next 12. The 80% range comes from that model's own past forecast errors.
5. **Evaluation:** MAE, RMSE, sMAPE, MASE and bias. The last 12 months are held out as a final test.

Finding: the model that wins the backtests is not always the best on the final 12 months, because the series changed behaviour around 2020. The website says so and shows every model, so the forecast should be read as a range.
