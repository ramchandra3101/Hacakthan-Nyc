import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import weaviate
from weaviate.auth import AuthApiKey
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from openai import OpenAI
import opik
from opik.integrations.openai import track_openai

# Initialize FastAPI app
app = FastAPI(title="Music Recommendation System")

# Configuration for Weaviate and Spotify
WEAVIATE_CLUSTER_URL = os.getenv('WEAVIATE_CLUSTER_URL') or 'https://your-weaviate-cluster-url.weaviate.cloud'
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY') or 'your-weaviate-api-key'

SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID') or 'your-spotify-client-id'
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET') or 'your-spotify-client-secret'

# Configure Opik for tracing
opik.configure(api_key=os.getenv('OPIK_API_KEY'))  # Use Opik API key from environment
os.environ["OPIK_PROJECT_NAME"] = "music-recommendation-system"

# Models for request and response
class MusicQuery(BaseModel):
    query: str

class Track(BaseModel):
    title: str
    artist: str
    mood: Optional[str] = None
    genre: Optional[str] = None

class MusicRecommendationResponse(BaseModel):
    query: str
    recommendations: List[Track]

# Initialize Spotify client
def get_spotify_client():
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                                client_secret=SPOTIPY_CLIENT_SECRET))
    return sp

# Initialize Weaviate client
def get_weaviate_client():
    try:
        # Properly initialize the Weaviate client
        client = weaviate.Client(
            url=os.getenv("WEAVIATE_CLUSTER_URL"),  # Set Weaviate URL in environment variable
            auth_client_secret=AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))  # Use the API key for authentication
        )
        return client
    except Exception as e:
        raise Exception(f"Failed to connect to Weaviate: {str(e)}")

# Initialize Friendli client
def get_friendli_client():
    friendli_client = OpenAI(
        base_url="https://api.friendli.ai/serverless/v1",
        api_key=os.environ.get('FRIENDLI_TOKEN')
    )
    return friendli_client

# Check if the Music collection exists in Weaviate, create if not
def setup_weaviate(client):
    try:
        # Check if the collection exists
        try:
            music_collection = client.schema.get("Music")
            print("Music collection already exists")
        except Exception:
            # Create the collection
            music_collection = client.schema.create_class({
                "class": "Music",
                "properties": [
                    {"name": "title", "dataType": ["text"]},
                    {"name": "artist", "dataType": ["text"]},
                    {"name": "genre", "dataType": ["text"]},
                    {"name": "mood", "dataType": ["text"]}
                ],
                "vectorizer": "text2vec-openai"
            })
            print("Created Music collection")
            
            # Add some sample music data
            sample_tracks = [
                {"title": "Lose Yourself", "artist": "Eminem", "genre": "Hip-Hop", "mood": "motivational"},
                {"title": "Eye of the Tiger", "artist": "Survivor", "genre": "Rock", "mood": "empowering"},
                {"title": "Happy", "artist": "Pharrell Williams", "genre": "Pop", "mood": "cheerful"},
                {"title": "Beautiful Day", "artist": "U2", "genre": "Rock", "mood": "uplifting"},
                {"title": "Don't Stop Believin'", "artist": "Journey", "genre": "Rock", "mood": "hopeful"}
            ]
            
            for track in sample_tracks:
                music_collection.add_data_object(track, class_name="Music")
                
        return music_collection
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set up Weaviate collection: {str(e)}")

@opik.track
def retrieve_context(client, user_query):
    """Retrieve relevant tracks from Weaviate based on the user query"""
    music_collection = client.data_object.get(class_name="Music")
    response = music_collection.query.near_text(
        query=user_query,
        limit=3
    )
    
    recommended_tracks = []
    for track in response:
        recommended_tracks.append({
            "title": track.properties.get("title"),
            "artist": track.properties.get("artist"),
            "genre": track.properties.get("genre"),
            "mood": track.properties.get("mood")
        })
    return recommended_tracks

@opik.track
def call_llm(client, messages):
    """Call the Friendli LLM to generate recommendations"""
    response = client.chat.completions.create(
      model="meta-llama-3.3-70b-instruct",
      messages=messages
    )
    return response

@opik.track
def generate_response(friendli_client, user_query, recommended_tracks):
    """Generate a response based on the user query and recommended tracks"""
    prompt = f"""
    You're a helpful music recommendation assistant. Reply to a user message for someone inquiring for
    music recommendations. The user query was: {user_query}

    These were the tracks that were extracted from the vector search:

    {recommended_tracks}
    
    Provide a thoughtful response that explains why these songs might help with their current situation,
    and format your response in a friendly, empathetic way.
    """

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = call_llm(friendli_client, messages)
    return response.choices[0].message.content

@opik.track(name="music-recommendation-flow")
def recommendation_flow(user_query):
    """The full recommendation flow, tracked with Opik"""
    # Get clients
    weaviate_client = get_weaviate_client()
    friendli_client = get_friendli_client()
    spotify_client = get_spotify_client()
    
    # Set up Weaviate if needed
    music_collection = setup_weaviate(weaviate_client)
    
    # Retrieve context from Weaviate
    context = retrieve_context(weaviate_client, user_query)
    
    # Generate response with Friendli
    response_text = generate_response(friendli_client, user_query, context)
    
    # Use Spotify to fetch real songs based on the context (genre or mood)
    genre = context[0]["genre"] if context else "Pop"
    sp_results = spotify_client.search(q=genre, type="track", limit=3)
    spotify_tracks = []
    for track in sp_results['tracks']['items']:
        spotify_tracks.append({
            "title": track['name'],
            "artist": track['artists'][0]['name'],
            "uri": track['uri']
        })

    return {
        "query": user_query,
        "recommendations": context,
        "response": response_text,
        "spotify_tracks": spotify_tracks
    }

@app.post("/recommend", response_model=MusicRecommendationResponse)
async def recommend_music(query: MusicQuery):
    """Endpoint to get music recommendations based on user query"""
    result = recommendation_flow(query.query)
    return {
        "query": query.query,
        "recommendations": result["recommendations"],
        "response": result["response"],
        "spotify_tracks": result["spotify_tracks"]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Music Recommendation API! Use /recommend to get music recommendations."}

# Startup event to ensure environment is set up correctly
@app.on_event("startup")
async def startup_event():
    # Check for required environment variables
    if not os.environ.get("FRIENDLI_TOKEN"):
        print("FRIENDLI_TOKEN not found in environment, please set it to use the API")
    
    # Test Weaviate connection
    try:
        client = get_weaviate_client()
        setup_weaviate(client)
        print("Weaviate connection successful and collection set up")
    except Exception as e:
        print(f"Warning: Could not initialize Weaviate: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
