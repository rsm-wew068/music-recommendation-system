import streamlit as st
import os
import json
import torch
import numpy as np
from recommender.wrmf import WRMF, build_interaction_samples
from recommender.embeddings import get_embeddings, get_average_audio_embedding
from recommender.recommend import generate_recommendations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/727/727245.png", width=80)
    st.title("🎶 Music Recommender")
    st.markdown("""
    <span style='color:#888'>Powered by Collaborative Filtering & Audio Features</span>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.info("Select a playlist and try both recommenders!", icon="💡")

# Main Title
st.markdown("""
    <h1 style='text-align:center; color:#4F8BF9;'>Music Recommendation System</h1>
    <p style='text-align:center; color:#888;'>Discover new tracks based on your taste!</p>
    """, unsafe_allow_html=True)

# Load playlist data
train_path = 'homework5/train_playlists.json'
test_path = 'homework5/test_playlists.json'
embedding_dir = 'homework5/audio_embeddings'
audio_clip_dir = 'homework5/audio_clips'  # Optional: directory for audio clips
cover_art_dir = 'homework5/cover_art'  # Optional: directory for cover art images

@st.cache_data
def load_playlists():
    with open(train_path) as f:
        train_playlists = json.load(f)
    with open(test_path) as f:
        test_playlists = json.load(f)
    return train_playlists, test_playlists

@st.cache_data
def load_embeddings():
    return get_embeddings(embedding_dir)

@st.cache_resource
def load_model(num_users, num_items, num_factors=32):
    model = WRMF(num_users, num_items, num_factors)
    # In a real app, load model weights here
    return model

train_playlists, test_playlists = load_playlists()
embedding_matrix, tids = load_embeddings()

if embedding_matrix.shape[0] == 0:
    st.error("No audio embeddings found. Please check that 'homework5/audio_embeddings/' contains .npy files.")
    st.stop()

_, tid_to_idx, idx_to_tid, tid_to_meta = build_interaction_samples(train_playlists)

playlist_options = {str(idx): plist for idx, plist in test_playlists.items()}
playlist_idx = st.selectbox('🎧 Select a test playlist', list(playlist_options.keys()))
playlist = playlist_options[playlist_idx]
playlist_tids = [track['tid'] for track in playlist if track['tid'] in tid_to_idx]

missing_tids = [track['tid'] for track in playlist if track['tid'] not in tids]
if missing_tids:
    st.warning(f"Some tracks in this playlist are missing audio embeddings and will be ignored: {missing_tids}")

# Playlist display
st.markdown("---")
st.subheader("Your Playlist")
cols = st.columns(2)
for i, track in enumerate(playlist):
    with cols[i % 2]:
        st.markdown(f"**{track['track_name']}** by *{track['artist_name']}*")
        # Show cover art if available
        cover_path = f"{cover_art_dir}/{track['tid']}.jpg"
        if os.path.exists(cover_path):
            st.image(cover_path, width=120)
        audio_path = f"{audio_clip_dir}/{track['tid']}.mp3"
        if os.path.exists(audio_path):
            st.audio(audio_path)

# Ensure log is always initialized
if 'log' not in st.session_state:
    st.session_state['log'] = []

# Tabs for recommenders
st.markdown("---")
tabs = st.tabs(["🎲 Collaborative Filtering", "🎵 Audio Similarity"])

with tabs[0]:
    st.subheader('Top 10 Recommendations (CF)')
    if st.button('Recommend (Collaborative Filtering)', key='cf'):
        try:
            model = load_model(len(train_playlists), len(tid_to_idx))
            all_item_embeddings = model.item_factors.weight.data
            _, recs = generate_recommendations(model, playlist_tids, all_item_embeddings, idx_to_tid, tid_to_idx, 10)
            with st.expander("See Recommended Tracks", expanded=True):
                for tid in recs:
                    artist, track = tid_to_meta.get(tid, ("Unknown", tid))
                    st.markdown(f"**{track}** by *{artist}* :blue_heart:")
                    cover_path = f"{cover_art_dir}/{tid}.jpg"
                    if os.path.exists(cover_path):
                        st.image(cover_path, width=120)
                    audio_path = f"{audio_clip_dir}/{tid}.mp3"
                    if os.path.exists(audio_path):
                        st.audio(audio_path)
                    else:
                        st.info("Audio clip not available.")
            st.session_state['log'].append({
                'playlist': playlist_idx,
                'method': 'Collaborative Filtering',
                'recommendations': recs
            })
        except Exception as e:
            st.error(f"Collaborative Filtering recommendation failed: {e}")

with tabs[1]:
    st.subheader('Top 10 Recommendations (Audio)')
    if st.button('Recommend (Audio Similarity)', key='audio'):
        try:
            playlist_embedding = get_average_audio_embedding(playlist_tids, embedding_dir)
            from recommender.recommend import get_similarity
            if embedding_matrix.shape[0] == 0:
                st.error("No audio embeddings available for similarity computation.")
            else:
                _, ranked_tids = get_similarity(playlist_tids, playlist_embedding, embedding_matrix, tids)
                with st.expander("See Recommended Tracks", expanded=True):
                    for tid in ranked_tids[:10]:
                        artist, track = tid_to_meta.get(tid, ("Unknown", tid))
                        st.markdown(f"**{track}** by *{artist}* :green_heart:")
                        cover_path = f"{cover_art_dir}/{tid}.jpg"
                        if os.path.exists(cover_path):
                            st.image(cover_path, width=120)
                        audio_path = f"{audio_clip_dir}/{tid}.mp3"
                        if os.path.exists(audio_path):
                            st.audio(audio_path)
                        else:
                            st.info("Audio clip not available.")
                st.session_state['log'].append({
                    'playlist': playlist_idx,
                    'method': 'Audio Similarity',
                    'recommendations': ranked_tids[:10]
                })
        except Exception as e:
            st.error(f"Audio Similarity recommendation failed: {e}")

# Show log (for metrics/debugging)
with st.expander('Show Recommendation Log'):
    st.write(st.session_state['log'])
