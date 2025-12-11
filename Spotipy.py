import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

auth_manager = SpotifyClientCredentials('b7a0830d30f343589d4508df4a0ebf94','413ced52bc834d289270f9d01bd86922')
sp = spotipy.Spotify(auth_manager=auth_manager)

def getTrackIDs(user, playlist_id):
    track_ids = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        track_ids.append(track['id'])
    return track_ids

def getTrackFeatures(id):
    track_info = sp.track(id)

    name = track_info['name']
    album = track_info['album']['name']
    artist = track_info['album']['artists'][0]['name']
    # release_date = track_info['album']['release_date']
    # length = track_info['duration_ms']
    # popularity = track_info['popularity']

    track_data = [name, album, artist] #, release_date, length, popularity
    return track_data

# Code for creating dataframe of feteched playlist

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist={0:"0l9dAmBrUJLylii66JOsHB?si=e1d97b8404e34343",1:"1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b ",2:"4cllEPvFdoX6NIVWPKai9I?si=dfa422af2e8448ef",3:"0deORnapZgrxFY4nsKr9JA?si=7a5aba992ea14c93",4:"4kvSlabrnfRCQWfN0MgtgA?si=b36add73b4a74b3a",5:"1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b",6:"37i9dQZEVXbMDoHDwVN2tF?si=c09391805b6c4651"}


