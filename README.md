# Spotify Audio-Based Recommender System

A content-based music recommender that uses **audio features only** without personalization.

## Stack

`Python` · `scikit-learn` · `Keras` · `pandas` · `numpy` · Jupyter Notebook
---

## Motivation

Spotify's recommendation engine (Discovery Weekly, Radio, Playlist Extension) is heavily tied to listening history. This creates two problems for discovery:

1. Modifying your "Preference" requires weeks of retraining the SPotify algorithm.
2. This approach reamplifies what the User likes and has a high probability to avoid new things.
It does satisfy the User more and is effective for personalized generation to incentivize long term User engagement. 
But for discovery purposes this is bad, since it covers less “ground” and has the tendency to stay on certain areas of variance.


> **Model vs. User distinction:** There's an important distinction between what the model is optimizing for and what the user actually wants. The model finds close neighbors in audio feature space to increase the probability of finding songs Users end up liking. But what users typically like may include: 
- songs that are adjacent in mood, 
- songs with different mood but similar audio features,
- or songs they'd enjoy even if they sound nothing like the query.


---

## Dataset

- **Source:** Open-source Kaggle dataset (Spotify deprecated its audio feature API)
- **Size:** ~132,000 songs, 1920–2020
- **Features:** `energy`, `loudness`, `acousticness`, `danceability`, `valence`, `tempo`, `instrumentalness`, `liveness`, `speechiness`, `mode`, `key`

---

## Architecture

### Three-Stage Funnel
1. **Artist filter** — exclude same-artist tracks
2. **Mode filter** — exclude songs in a different mode (major/minor) than the query
3. **Similarity ranking** — rank remaining songs by one of three algorithms

### Feature Selection
Low-variance features (`liveness`, `speechiness`, `key`) were dropped. 

PCA on the remaining features confirmed that the final 3-feature set (`energy`, `loudness`, `acousticness`) is non-redundant.
```
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)
print(pca.explained_variance_ratio_)

[0.80708605 0.13898374]
```
While `mode` is used as a filter only, not included in the similarity vector.

---

## Models

### Cosine Similarity (Baseline)
Measures the angle between two songs in feature space. A score of 0.99 means near-identical audio profile direction, regardless of magnitude.

### KNN — K-Nearest Neighbors
Measures absolute Euclidean distance. More sensitive to feature magnitude than ratios. Score formula: `1 / (1 + d)`.

```
NearestNeighbors(n_neighbors=500, metric='euclidean', algorithm='ball_tree')
```

### Autoencoder + KNN
Compresses 3 audio features into a 2-dimensional latent space (encoder → bottleneck → decoder, trained with MSE loss). KNN then runs in the learned latent space rather than raw feature space.

```
Architecture: 3 → 16 → 2 → 16 → 3
Loss: MSE | Epochs: 50 | Batch: 256
```

---

## Evaluation Metrics
Vector distance alone isn't sufficient as a final metric — there's a gap between how audio features are numerically designed and how humans judge two songs as "similar." We need a second metric - User's feedback, but it takes a lot of time and resources.

**M1 — Query-to-Rec Similarity:** Average similarity score across top-10 recommendations.

**M2 — Likeability Rate:** Out of 10 recommendations, how many are added to the user's liked playlist. Evaluated personally due to limitations.

---

## Results
**Similarity grading (used in evaluation tables):**
`S` = near-identical | `A` = very similar | `B` = noticeably similar | `C` = minor | `D` = weak | `E` = barely noticeable | `F` = no resemblance
### Cosine Similarity

**Re:Re: — ASIAN KUNG-FU GENERATION**

| Simi | Track | Like |
|------|-------|------|
| S | The Velvet Underground — Caroline | ✓ |
| B | Tom Petty and the Heartbreakers — Even The Losers | ✓ |
| B- | Earshot — Get Away | ✓ |
| S | Blake Shelton — Honey Bee | |
| E | Lifehouse — Falling In | |
| F | David Banner, Lil Flip — Like A Pimp | |

**Me and Your Mama — Childish Gambino** *(a hard query — psychedelic soul is an ambiguous genre itself + song complexity)*

| Simi | Track | Like |
|------|-------|------|
| A- | Aerosmith — Reefer Head Woman | ✓ |
| A- | Bo Diddley — Goin' Down Slow | ✓ |
| B | Carlos Santana — One With You | ✓ |
| F | 2 LIVE CREW — Put Her In The Buck | |
| F | "Weird Al" Yankovic — Word Crimes | |

**Result:** ~2–3 / 10 likeable. ~2–3 / 10 genuinely similar.

---

### KNN

**Notable finding:** Several recommendations persisted across both Cosine Sim and KNN despite low similarity scores (e.g., Lifehouse — Falling In, ). When the same result appears regardless of algorithm, it suggests a **data issue** — those songs likely have extreme or misleading feature values that place them near many query songs in any feature space.

KNN performed worse on similarity scores but slightly better on likeability for the Childish Gambino query.

---

### Autoencoder + KNN

**Re:Re: — ASIAN KUNG-FU GENERATION**

| Simi | Track | Like |
|------|-------|------|
| A | The Sonics — Boss Hoss | |
| B- | Grouplove — Lovely Cup | |
| B- | Earshot — Get Away | ✓ |
| F | Lil Skies — Riot | |
| F | Rick Ross, JAY-Z — Free Mason | ✓ |

Worst results of the three models in both similarity and likeability. The model is blind to anything outside audio features — including the user's broader taste context: Me personally liking Free Mason (hiphop genre) despite F similarity

---

## Model Comparison

| Metric | Cosine Similarity | KNN | Autoencoder + KNN |
|--------|------------------|-----|-------------------|
| Similarity quality | **Best** | Worse | Worst |
| Likeability | ~2–3/10 | Slightly higher (Gambino) | Lowest |
| Persistent bad recs | Yes | Yes | Yes |

---

## Limitations

- **No genre data:** Genre-based filtering (like the existing mode filter) would improve results significantly. Current open-source datasets don't include genre labels.
- **Single evaluator:** Manual evaluation introduces personal bias. Proper evaluation would require multiple listeners.

---

## Future Work

- **Genre filter** — filter out same-genre recommendations for cross-genre discovery.
- **Popularity sort** — discover songs/artists with low listening count; niche discovery Spotify doesn't prioritize.

---

