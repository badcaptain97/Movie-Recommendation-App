import joblib
from pathlib import Path

# Get the path to the current file
current_file = Path(__file__).resolve()
base_dir = current_file.parent  # Directory containing the current file

# Load the files relative to the file location
df = joblib.load(base_dir / 'data.pkl')
cos_sim = joblib.load(base_dir / 'cos_sim.pkl')


# final
def recommend_movies(movie_name, top_n=5):
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return "Movie unavailable"

    idx = idx[0]

    sim_score = list(enumerate(cos_sim[idx]))
    # print(sim_score)
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:top_n + 1]

    movie_id = [i[0] for i in sim_score]

    return df[['title']].iloc[movie_id]