import streamlit as st
import re
import ast
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import DB_CONFIG, GOOGLE_API_KEY, RAWG_API_KEY, MAIN_USER
import os
import pandas as pd
import random
from datetime import datetime
import sqlite3

# Import authentication system
from auth_system import initialize_auth, require_auth, show_auth_interface

# NEW: Import VectorDB Manager
from vector_db_manager import VectorDBManager

# Configure page
st.set_page_config(
    page_title="Game Assistant",
    page_icon="🎮",
    layout="wide"
)


# ---------------------------------------------------------------------
# 🧠 VECTORDB ENHANCEMENTS - NEW FUNCTIONS
# ---------------------------------------------------------------------
def initialize_vectordb():
    """Initialize VectorDB manager"""
    try:
        if 'vectordb' not in st.session_state:
            st.session_state.vectordb = VectorDBManager()
        return st.session_state.vectordb
    except Exception as e:
        st.error(f"❌ VectorDB initialization failed: {e}")
        return None


def get_semantic_recommendations(user_games, user_prefs, n_results=6):
    """Get semantic recommendations using VectorDB"""
    vectordb = initialize_vectordb()
    if not vectordb:
        return []

    try:
        all_recommendations = []

        # Strategy 1: Semantic search based on user's games
        for game_name, genre in user_games[:3]:
            search_query = f"{game_name} {genre} similar games"

            similar_games = vectordb.semantic_search(
                query=search_query,
                n_results=4,
                genre_filter=genre.lower() if genre else None
            )

            # Convert VectorDB results to game format
            for similar_game in similar_games:
                game_info = {
                    "name": similar_game['name'],
                    "genres": similar_game['genres'].split(', ') if similar_game.get('genres') else [],
                    "platforms": similar_game['platforms'].split(', ') if similar_game.get('platforms') else [],
                    "source": "semantic_search",
                    "similarity_score": 1 - (similar_game.get('distance', 0) if similar_game.get('distance') else 0),
                    "vectordb_enhanced": True
                }

                if not any(g['name'] == game_info['name'] for g in all_recommendations):
                    all_recommendations.append(game_info)

        # Strategy 2: Favorite genres semantic search
        favorite_genres = user_prefs.get('favorite_genres', [])
        for genre in favorite_genres[:2]:
            search_query = f"best {genre} games highly rated"

            genre_games = vectordb.semantic_search(
                query=search_query,
                n_results=3,
                genre_filter=genre.lower()
            )

            for game in genre_games:
                game_info = {
                    "name": game['name'],
                    "genres": game['genres'].split(', ') if game.get('genres') else [],
                    "platforms": game['platforms'].split(', ') if game.get('platforms') else [],
                    "source": "favorite_genre_semantic",
                    "similarity_score": 1 - (game.get('distance', 0) if game.get('distance') else 0),
                    "vectordb_enhanced": True
                }

                if not any(g['name'] == game_info['name'] for g in all_recommendations):
                    all_recommendations.append(game_info)

        # Remove duplicates and sort by similarity
        unique_recommendations = []
        seen_names = set()

        for rec in all_recommendations:
            if rec['name'] not in seen_names:
                seen_names.add(rec['name'])
                unique_recommendations.append(rec)

        # Sort by similarity score
        unique_recommendations.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        return unique_recommendations[:n_results]

    except Exception as e:
        print(f"Semantic recommendations error: {e}")
        return []


def enhance_semantic_recommendation(semantic_rec):
    """Enhance semantic recommendation with RAWG API data"""
    try:
        # Search for the game in RAWG API to get detailed info
        headers = {"User-Agent": "VideoGameRAG/1.0", "Accept": "application/json"}

        response = requests.get(
            "https://api.rawg.io/api/games",
            params={
                "key": RAWG_API_KEY,
                "search": semantic_rec['name'],
                "page_size": 1,
                "search_precise": True
            },
            headers=headers,
            timeout=5
        )

        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                # Use the RAWG game data but keep semantic info
                rawg_game = results[0]
                enhanced_rec = get_detailed_game_info(rawg_game)
                enhanced_rec['source'] = semantic_rec.get('source', 'semantic')
                enhanced_rec['similarity_score'] = semantic_rec.get('similarity_score', 0)
                enhanced_rec['vectordb_enhanced'] = True
                return enhanced_rec

        # If no RAWG data, return the semantic recommendation as-is
        semantic_rec['vectordb_enhanced'] = True
        return semantic_rec

    except Exception as e:
        print(f"Error enhancing semantic recommendation: {e}")
        semantic_rec['vectordb_enhanced'] = True
        return semantic_rec


def get_enhanced_standard_user_recommendations(user_games):
    """Enhanced recommendations combining RAWG API + VectorDB semantic search"""
    # Get traditional RAWG recommendations
    rawg_recommendations = get_standard_user_recommendations(user_games)

    # Get VectorDB semantic recommendations
    user_prefs = st.session_state.get('user_preferences', {})
    semantic_recommendations = get_semantic_recommendations(user_games, user_prefs, n_results=6)

    # Combine and prioritize
    all_recommendations = []

    # Add VectorDB recommendations first (enhance with RAWG data)
    for sem_rec in semantic_recommendations:
        enhanced_rec = enhance_semantic_recommendation(sem_rec)
        if enhanced_rec:
            all_recommendations.append(enhanced_rec)

    # Add RAWG recommendations (avoid duplicates)
    for rawg_rec in rawg_recommendations:
        if not any(rec['name'] == rawg_rec['name'] for rec in all_recommendations):
            rawg_rec['vectordb_enhanced'] = False
            rawg_rec['source'] = 'rawg'
            all_recommendations.append(rawg_rec)

    # Remove duplicates and sort by quality
    unique_recommendations = []
    seen_names = set()

    for rec in all_recommendations:
        if rec['name'] not in seen_names:
            seen_names.add(rec['name'])
            unique_recommendations.append(rec)

    # Sort by rating/similarity
    unique_recommendations.sort(
        key=lambda x: (
            x.get('metacritic', 0) or 0,
            x.get('rating', 0) or 0,
            x.get('similarity_score', 0)
        ),
        reverse=True
    )

    return unique_recommendations[:10]


def populate_vectordb_from_rawg():
    """Populate VectorDB with popular games from RAWG API"""
    vectordb = initialize_vectordb()
    if not vectordb:
        return False

    try:
        headers = {"User-Agent": "VideoGameRAG/1.0", "Accept": "application/json"}

        # Get popular games from multiple genres
        popular_genres = ['action', 'rpg', 'adventure', 'strategy', 'shooter']

        for genre in popular_genres:
            params = {
                "key": RAWG_API_KEY,
                "genres": genre,
                "page_size": 20,
                "ordering": "-rating",
                "page": 1
            }

            response = requests.get(
                "https://api.rawg.io/api/games",
                params=params,
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                games = response.json().get("results", [])

                for game in games:
                    try:
                        # Add game to VectorDB
                        game_id = f"rawg_{game['id']}"
                        name = game.get('name', '')
                        description = game.get('description_raw', '') or "No description available"
                        genres = [g['name'] for g in game.get('genres', [])]
                        platforms = [p['platform']['name'] for p in game.get('platforms', [])]

                        # Additional metadata
                        metadata = {
                            "source": "rawg",
                            "rating": game.get('rating', 0),
                            "released": game.get('released', ''),
                            "metacritic": game.get('metacritic'),
                            "playtime": game.get('playtime', 0)
                        }

                        success = vectordb.add_game(
                            game_id=game_id,
                            name=name,
                            description=description,
                            genres=genres,
                            platforms=platforms,
                            metadata=metadata
                        )

                        if success:
                            print(f"✅ Added to VectorDB: {name}")

                    except Exception as e:
                        print(f"Error adding game {game.get('name')} to VectorDB: {e}")
                        continue

        st.success("🎯 VectorDB populated with popular games!")
        return True

    except Exception as e:
        st.error(f"Error populating VectorDB: {e}")
        return False


# ---------------------------------------------------------------------
# 🗄️ GENRE MAPPING (IMPROVED FOR BOTH SYSTEMS)
# ---------------------------------------------------------------------
GENRE_MAPPING = {
    'Lucha': ['fighting', 'arcade'],
    'Beat em up': ['fighting', 'action', 'beat-em-up'],
    'RPG': ['role-playing-games-rpg'],
    'JRPG': ['role-playing-games-rpg'],
    'ARPG': ['role-playing-games-rpg', 'action'],
    'MMORPG': ['massively-multiplayer', 'role-playing-games-rpg'],
    'Aventura': ['adventure'],
    'Accion-Aventura': ['action', 'adventure'],
    'Accion': ['action'],
    'Novela Visual': ['adventure', 'visual-novel'],
    'Plataformas': ['platformer'],
    'Metroidvania': ['platformer', 'adventure'],
    'TRPG': ['strategy', 'role-playing-games-rpg'],
    'Estrategia': ['strategy'],
    'Estrategia por turnos': ['strategy'],
    'RTS': ['strategy'],
    'Defensa de torres': ['strategy'],
    'Disparos en primera persona': ['shooter'],
    'Disparos en tercera persona': ['shooter', 'action'],
    'Disparos': ['shooter'],
    'Run and Gun': ['shooter', 'arcade'],
    'Hack and Slash': ['action'],
    'Survival Horror': ['adventure', 'action'],
    'Deportes': ['sports'],
    'Carreras': ['racing'],
    'Ritmo': ['music', 'arcade'],
    'Sandbox': ['action', 'adventure'],
    'Fiesta': ['family', 'arcade'],
    'Gacha': ['role-playing-games-rpg', 'strategy'],
    'Casual': ['casual', 'indie'],
    'Social': ['casual'],
    'default': ['action', 'adventure']
}

ENGLISH_TO_SPANISH_GENRES = {
    'fighting': 'Lucha', 'fight': 'Lucha', 'fighter': 'Lucha',
    'beat em up': 'Beat em up', 'brawler': 'Beat em up',
    'rpg': 'RPG', 'role playing': 'RPG', 'jrpg': 'JRPG',
    'action rpg': 'ARPG', 'mmorpg': 'MMORPG',
    'adventure': 'Aventura', 'action adventure': 'Accion-Aventura',
    'action': 'Accion', 'visual novel': 'Novela Visual',
    'platformer': 'Plataformas', 'platform': 'Plataformas',
    'metroidvania': 'Metroidvania', 'strategy': 'Estrategia',
    'tactical rpg': 'TRPG', 'turn based strategy': 'Estrategia por turnos',
    'real time strategy': 'RTS', 'tower defense': 'Defensa de torres',
    'fps': 'Disparos en primera persona', 'shooter': 'Disparos',
    'hack and slash': 'Hack and Slash', 'survival horror': 'Survival Horror',
    'horror': 'Survival Horror', 'sports': 'Deportes', 'racing': 'Carreras',
    'rhythm': 'Ritmo', 'music': 'Ritmo', 'sandbox': 'Sandbox',
    'party': 'Fiesta', 'gacha': 'Gacha', 'casual': 'Casual', 'social': 'Social'
}

# NEW: Direct English to RAWG mapping for standard users
ENGLISH_TO_RAWG_MAPPING = {
    'fighting': ['fighting', 'fighter', 'brawler'],
    'fight': ['fighting', 'fighter'],
    'fighter': ['fighting', 'fighter'],
    'beat em up': ['fighting', 'action', 'beat-em-up'],
    'brawler': ['fighting', 'action'],
    'rpg': ['role-playing-games-rpg'],
    'role playing': ['role-playing-games-rpg'],
    'jrpg': ['role-playing-games-rpg'],
    'action rpg': ['role-playing-games-rpg', 'action'],
    'mmorpg': ['massively-multiplayer', 'role-playing-games-rpg'],
    'adventure': ['adventure'],
    'action adventure': ['action', 'adventure'],
    'action': ['action'],
    'visual novel': ['adventure', 'visual-novel'],
    'platformer': ['platformer'],
    'platform': ['platformer'],
    'metroidvania': ['platformer', 'adventure'],
    'strategy': ['strategy'],
    'tactical rpg': ['strategy', 'role-playing-games-rpg'],
    'turn based strategy': ['strategy'],
    'real time strategy': ['strategy'],
    'tower defense': ['strategy'],
    'fps': ['shooter'],
    'shooter': ['shooter'],
    'hack and slash': ['action'],
    'survival horror': ['horror'],
    'horror': ['horror'],
    'sports': ['sports'],
    'racing': ['racing'],
    'rhythm': ['music'],
    'music': ['music'],
    'sandbox': ['action', 'adventure'],
    'party': ['family'],
    'gacha': ['role-playing-games-rpg', 'strategy'],
    'casual': ['casual', 'indie'],
    'social': ['casual'],
    'default': ['action', 'adventure']
}

# Platform mapping for RAWG API
PLATFORM_MAPPING = {
    'PC': [4, 5, 6],
    'PlayStation': [18, 16, 15, 27, 19, 17],
    'Xbox': [1, 14, 80],
    'Nintendo Switch': [7],
    'Mobile': [8, 9, 79],
    'Mac': [5]
}

# ---------------------------------------------------------------------
# 🧩 USER ROLE CONFIGURATION
# ---------------------------------------------------------------------
MAIN_USER = MAIN_USER  # Change this to your main username


def is_main_user():
    """Check if current user is the main user with PostgreSQL access"""
    return st.session_state.get('username') == MAIN_USER


def parse_sql_result(result):
    """Parse SQL result that comes as string representation of list/tuples"""
    if isinstance(result, str):
        try:
            # Try to parse string as Python data structure
            parsed = ast.literal_eval(result)
            return parsed
        except (SyntaxError, ValueError):
            # If literal_eval fails, try manual parsing for common formats
            if result.startswith('[') and result.endswith(']'):
                # Remove brackets and split by rows
                content = result[1:-1].strip()
                if content:
                    # Parse tuples within the list
                    rows = []
                    # Simple parsing for tuples like ('value1', 'value2')
                    tuple_pattern = r'\([^)]+\)'
                    matches = re.findall(tuple_pattern, content)
                    for match in matches:
                        # Remove parentheses and split by commas
                        tuple_content = match[1:-1].strip()
                        values = [v.strip().strip("'\"") for v in tuple_content.split(',')]
                        rows.append(tuple(values))
                    return rows if rows else [result]
            return [result]
    return result


def debug_postgresql_connection():
    """Debug function to check PostgreSQL connection and data - FIXED"""
    try:
        st.subheader("🔧 PostgreSQL Debug Information")

        # Test basic connection
        sql_chain, execute_tool, summary_chain, error = create_rag_sql_chain()
        if error:
            st.error(f"❌ Connection Error: {error}")
            return False

        st.success("✅ PostgreSQL connection successful!")

        # Check table structure
        st.markdown("### 📋 Table Structure")
        try:
            result = execute_tool.invoke({
                "query": "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'played_games'"
            })

            # PARSE THE RESULT
            parsed_result = parse_sql_result(result)
            st.write(f"🔍 Debug - Raw result: {result}")
            st.write(f"🔍 Debug - Parsed result: {parsed_result}")
            st.write(f"🔍 Debug - Parsed result type: {type(parsed_result)}")

            if parsed_result and isinstance(parsed_result, list):
                st.write("Table columns:")
                for col in parsed_result:
                    if isinstance(col, (list, tuple)) and len(col) >= 2:
                        st.write(f"- {col[0]} ({col[1]})")
                    else:
                        st.write(f"- Raw column data: {col}")
            else:
                st.warning("No columns found or unexpected result format")

        except Exception as e:
            st.error(f"Error getting table structure: {e}")

        # Check total game count
        st.markdown("### 📊 Game Counts")
        try:
            # Total games
            result = execute_tool.invoke({"query": "SELECT COUNT(*) FROM played_games"})

            # PARSE THE RESULT
            parsed_result = parse_sql_result(result)
            st.write(f"🔍 Debug - Raw count result: {result}")
            st.write(f"🔍 Debug - Parsed count result: {parsed_result}")
            st.write(f"🔍 Debug - Parsed result type: {type(parsed_result)}")

            if parsed_result and isinstance(parsed_result, list) and len(parsed_result) > 0:
                count_data = parsed_result[0]
                st.write(f"🔍 Debug - First element: {count_data}, type: {type(count_data)}")

                # Extract count from various possible formats
                if isinstance(count_data, (list, tuple)) and len(count_data) > 0:
                    raw_count = count_data[0]
                elif isinstance(count_data, (int, float)):
                    raw_count = count_data
                else:
                    raw_count = count_data

                st.write(f"🔍 Debug - Raw count: {raw_count}, type: {type(raw_count)}")

                # Convert to integer safely
                try:
                    if isinstance(raw_count, (int, float)):
                        total_games = int(raw_count)
                    else:
                        # Handle string or other types
                        total_games = int(str(raw_count).strip('[]()'))
                    st.success(f"**Total games in database:** {total_games}")
                except (ValueError, TypeError) as e:
                    st.error(f"Could not parse total games count: {raw_count} - Error: {e}")
                    total_games = 0
            else:
                total_games = 0
                st.warning("Empty or unexpected result from count query")

            if total_games > 0:
                # Games by genre
                result = execute_tool.invoke({
                    "query": "SELECT genero_principal, COUNT(*) FROM played_games GROUP BY genero_principal"
                })
                parsed_result = parse_sql_result(result)
                st.write(f"🔍 Debug - Parsed genre result: {parsed_result}")
                st.write("**Games by genre:**")

                if parsed_result and isinstance(parsed_result, list):
                    for item in parsed_result:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            genre = item[0]
                            count = item[1]
                            # Handle count format
                            if isinstance(count, (list, tuple)) and len(count) > 0:
                                count_display = count[0]
                            else:
                                count_display = count
                            st.write(f"- {genre}: {count_display} games")
                        else:
                            st.write(f"- Raw genre data: {item}")

                # Sample games
                result = execute_tool.invoke({
                    "query": "SELECT nombre, genero_principal, plataforma FROM played_games LIMIT 5"
                })
                parsed_result = parse_sql_result(result)
                st.write(f"🔍 Debug - Parsed sample games: {parsed_result}")
                st.write("**Sample games (first 5):**")

                if parsed_result and isinstance(parsed_result, list):
                    for game in parsed_result:
                        if isinstance(game, (list, tuple)) and len(game) >= 3:
                            st.write(f"- {game[0]} | Genre: {game[1]} | Platform: {game[2]}")
                        else:
                            st.write(f"- Raw game data: {game}")

            return total_games > 0

        except Exception as e:
            st.error(f"Error counting games: {e}")
            return False

    except Exception as e:
        st.error(f"Debug error: {e}")
        return False


def has_postgresql_games():
    """Check if PostgreSQL database has games - FIXED with parsing"""
    try:
        sql_chain, execute_tool, summary_chain, error = create_rag_sql_chain()
        if error:
            st.error(f"Database connection error: {error}")
            return False

        # Check if there are any games in the database
        result = execute_tool.invoke({"query": "SELECT COUNT(*) FROM played_games"})

        # PARSE THE RESULT
        parsed_result = parse_sql_result(result)
        st.write(f"🔍 Debug - Raw result: {result}")
        st.write(f"🔍 Debug - Parsed result: {parsed_result}")

        if parsed_result and isinstance(parsed_result, list) and len(parsed_result) > 0:
            count_data = parsed_result[0]
            st.write(f"🔍 Debug - Count data: {count_data}, type: {type(count_data)}")

            # Extract count from various formats
            if isinstance(count_data, (list, tuple)) and len(count_data) > 0:
                raw_count = count_data[0]
            elif isinstance(count_data, (int, float)):
                raw_count = count_data
            else:
                raw_count = count_data

            st.write(f"🔍 Debug - Extracted raw count: {raw_count}, type: {type(raw_count)}")

            # Convert to integer safely
            try:
                if isinstance(raw_count, (int, float)):
                    game_count = int(raw_count)
                else:
                    # Handle string format like "333" or "[333]" or "(333)"
                    game_count = int(str(raw_count).strip('[]()'))

                st.success(f"📊 Found {game_count} total games in PostgreSQL")
                return game_count > 0

            except (ValueError, TypeError) as e:
                st.error(f"Could not parse game count: {raw_count} - Error: {e}")
                return False
        else:
            st.warning("Unexpected result format from database")
            return False

    except Exception as e:
        st.error(f"Error checking PostgreSQL games: {e}")
        return False


# ---------------------------------------------------------------------
# 🧩 SQLITE DATABASE FOR STANDARD USERS
# ---------------------------------------------------------------------
def init_user_sqlite_db(username):
    """Initialize SQLite database for a standard user"""
    db_path = f"user_games_{username}.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create user's personal games table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            platform TEXT NOT NULL,
            status TEXT NOT NULL,
            genre TEXT NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    return db_path


def get_user_sqlite_db(username):
    """Get SQLite database connection for a standard user"""
    db_path = f"user_games_{username}.db"
    # Initialize database if it doesn't exist
    init_user_sqlite_db(username)
    return sqlite3.connect(db_path)


def add_game_to_sqlite(username, name, platform, status, genre, notes=""):
    """Add a game to user's SQLite database"""
    try:
        conn = get_user_sqlite_db(username)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO user_games (name, platform, status, genre, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, platform, status, genre, notes))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding game to SQLite: {e}")
        return False


def get_user_games_count(username):
    """Get count of games in user's SQLite database"""
    try:
        conn = get_user_sqlite_db(username)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM user_games')
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else 0
    except:
        return 0


def get_user_games(username):
    """Get all games from user's SQLite database"""
    try:
        conn = get_user_sqlite_db(username)
        cursor = conn.cursor()

        cursor.execute('SELECT id, name, platform, status, genre, notes FROM user_games ORDER BY name')
        games = cursor.fetchall()
        conn.close()

        return games
    except:
        return []


def get_user_games_by_status(username, status):
    """Get user's games by status"""
    try:
        conn = get_user_sqlite_db(username)
        cursor = conn.cursor()

        cursor.execute('SELECT name, genre FROM user_games WHERE status = ?', (status,))
        games = cursor.fetchall()
        conn.close()

        return games
    except:
        return []


def delete_user_game(username, game_id):
    """Delete a game from user's SQLite database"""
    try:
        conn = get_user_sqlite_db(username)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM user_games WHERE id = ?', (game_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting game: {e}")
        return False


# ---------------------------------------------------------------------
# 🧩 AUTHENTICATION CHECK
# ---------------------------------------------------------------------
def check_authentication():
    """Check if user is authenticated, show auth interface if not"""
    auth_system = initialize_auth()

    if 'user_session' not in st.session_state or not auth_system.validate_session(st.session_state.user_session):
        show_auth_interface()
        st.stop()

    # User is authenticated, continue with main app
    return True


# ---------------------------------------------------------------------
# 🧩 USER PROFILE MANAGEMENT
# ---------------------------------------------------------------------
@require_auth()
def show_user_profile():
    """Show user profile and preferences"""
    auth_system = initialize_auth()

    st.subheader("👤 User Profile")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Username", st.session_state.username)
        if is_main_user():
            st.success("🎮 PostgreSQL Library Access")
            # Show PostgreSQL game count
            postgres_count = has_postgresql_games()
            if postgres_count:
                st.metric("PostgreSQL Games", "Available")
            else:
                st.metric("PostgreSQL Games", "Empty")
        else:
            st.info("👤 Personal SQLite Library")
            # Show SQLite game count
            sqlite_count = get_user_games_count(st.session_state.username)
            st.metric("My Games", sqlite_count)

        if 'user_preferences' in st.session_state:
            prefs = st.session_state.user_preferences
            st.metric("Favorite Genres", len(prefs.get('favorite_genres', [])))
            st.metric("Rating Threshold", prefs.get('rating_threshold', 3.5))

    with col2:
        with st.expander("🎯 Game Preferences"):
            # Ensure user_preferences exists in session state
            if 'user_preferences' not in st.session_state:
                st.session_state.user_preferences = auth_system.get_user_preferences(st.session_state.username)

            # Favorite genres selection - Use English genres for both user types
            available_genres = sorted(list(ENGLISH_TO_RAWG_MAPPING.keys()))
            current_favorites = st.session_state.user_preferences.get('favorite_genres', [])

            favorite_genres = st.multiselect(
                "Favorite Genres",
                options=available_genres,
                default=current_favorites,
                help="Select your favorite game genres"
            )

            # Rating threshold
            rating_threshold = st.slider(
                "Minimum Rating Threshold",
                min_value=0.0,
                max_value=5.0,
                value=float(st.session_state.user_preferences.get('rating_threshold', 3.5)),
                step=0.1,
                help="Only show games with ratings above this threshold"
            )

            # Platforms preference
            platforms = ["PC", "PlayStation", "Xbox", "Nintendo Switch", "Mobile", "Mac"]
            current_platforms = st.session_state.user_preferences.get('preferred_platforms', [])

            preferred_platforms = st.multiselect(
                "Preferred Platforms",
                options=platforms,
                default=current_platforms,
                help="Select platforms you own or prefer"
            )

            # Recommendation variety slider
            variety_level = st.slider(
                "Recommendation Variety",
                min_value=1,
                max_value=5,
                value=st.session_state.user_preferences.get('variety_level', 3),
                help="1 = Very similar games, 5 = More diverse recommendations"
            )

            if st.button("Save Preferences"):
                new_preferences = {
                    'favorite_genres': favorite_genres,
                    'preferred_platforms': preferred_platforms,
                    'rating_threshold': rating_threshold,
                    'variety_level': variety_level
                }

                if auth_system.update_user_preferences(st.session_state.username, new_preferences):
                    st.session_state.user_preferences = new_preferences
                    st.success("Preferences saved successfully!")
                else:
                    st.error("Failed to save preferences")

        if st.button("🚪 Logout", type="secondary"):
            if auth_system.logout_user(st.session_state.user_session):
                for key in ['user_session', 'username', 'user_preferences']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Logged out successfully!")
                st.rerun()


# ---------------------------------------------------------------------
# 🧩 ENHANCED RECOMMENDATIONS INTERFACE
# ---------------------------------------------------------------------
@require_auth()
def show_enhanced_standard_user_recommendations():
    """Enhanced recommendations for standard users with VectorDB"""
    st.markdown("### 🧠 Enhanced Personalized Recommendations")

    # Check if user has games in their library
    user_game_count = get_user_games_count(st.session_state.username)

    if user_game_count == 0:
        st.warning("📚 You need to add at least one game to your library to get recommendations!")
        st.info("💡 Go to the 'Add Game' tab to add some games you've played or want to play.")
        return

    st.success(f"🎮 Found {user_game_count} games in your library!")

    # Initialize VectorDB
    vectordb = initialize_vectordb()
    vectordb_available = vectordb is not None

    # Let user choose which games to base recommendations on
    st.markdown("#### Select games to base recommendations on:")

    # Get user's games by status
    completed_games = get_user_games_by_status(st.session_state.username, "Completed")
    playing_games = get_user_games_by_status(st.session_state.username, "Playing")

    all_user_games = completed_games + playing_games

    if not all_user_games:
        st.warning("Add some 'Completed' or 'Playing' games to get better recommendations!")
        all_user_games = get_user_games(st.session_state.username)
        if all_user_games:
            all_user_games = [(game[1], game[4]) for game in all_user_games]  # (name, genre)

    if all_user_games:
        # Let user select games for recommendations
        game_options = [f"{game[0]} ({game[1]})" for game in all_user_games]
        selected_game_indices = st.multiselect(
            "Choose games you like (select 1-3 for best results):",
            options=range(len(game_options)),
            format_func=lambda x: game_options[x],
            max_selections=3
        )

        if selected_game_indices:
            selected_games = [all_user_games[i] for i in selected_game_indices]
            game_titles = [game[0] for game in selected_games]
            game_genres = [game[1] for game in selected_games]

            st.info(f"🎯 Getting recommendations based on: {', '.join(game_titles)}")
            st.info(f"🎭 Selected genres: {', '.join(set(game_genres))}")

            # Recommendation type selector
            if vectordb_available:
                rec_type = st.radio(
                    "Recommendation Type:",
                    ["Enhanced (VectorDB + RAWG)", "Traditional (RAWG only)", "Semantic (VectorDB only)"],
                    index=0
                )
            else:
                rec_type = "Traditional (RAWG only)"
                st.info("🧠 VectorDB not available. Using traditional recommendations.")

            if st.button("🎮 Get Recommendations", type="primary"):
                with st.spinner("Finding enhanced recommendations..."):
                    if rec_type == "Enhanced (VectorDB + RAWG)":
                        recommendations = get_enhanced_standard_user_recommendations(selected_games)
                    elif rec_type == "Traditional (RAWG only)":
                        recommendations = get_standard_user_recommendations(selected_games)
                    else:  # Semantic only
                        user_prefs = st.session_state.get('user_preferences', {})
                        semantic_recs = get_semantic_recommendations(selected_games, user_prefs)
                        recommendations = [enhance_semantic_recommendation(rec) for rec in semantic_recs]

                display_enhanced_recommendations(recommendations)
        else:
            st.info("👆 Select 1-3 games from your library to get personalized recommendations")
    else:
        st.warning("No games found in your library. Add some games first!")


def display_enhanced_recommendations(recommendations):
    """Display recommendations with VectorDB enhancements"""
    if not recommendations:
        st.warning("No recommendations found.")
        return

    st.subheader("🎮 Enhanced Recommendations")

    # Show recommendation sources
    semantic_count = sum(1 for r in recommendations if
                         r.get('source', '').startswith('semantic') or r.get('source', '').startswith('favorite'))
    rawg_count = sum(1 for r in recommendations if
                     not r.get('source', '').startswith('semantic') and not r.get('source', '').startswith('favorite'))
    vectordb_count = sum(1 for r in recommendations if r.get('vectordb_enhanced', False))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Semantic Matches", semantic_count)
    with col2:
        st.metric("RAWG Matches", rawg_count)
    with col3:
        st.metric("VectorDB Enhanced", vectordb_count)

    # Show recommendations in columns
    cols = st.columns(2)

    for idx, game in enumerate(recommendations):
        with cols[idx % 2]:
            with st.container():
                st.markdown("---")

                # Header with enhancement badge
                col1, col2 = st.columns([1, 2])

                with col1:
                    if game.get("background_image"):
                        st.image(game["background_image"], width=80)
                    else:
                        st.image("🎮", width=80)

                with col2:
                    # Show enhancement badge
                    if game.get('vectordb_enhanced'):
                        st.markdown(f"### {game['name']} 🧠")
                    else:
                        st.markdown(f"### {game['name']}")

                    # Show source badge
                    source = game.get('source', 'rawg')
                    if source.startswith('semantic'):
                        st.caption("🔍 Semantic Match")
                    elif source == 'favorite_genre_semantic':
                        st.caption("❤️ Favorite Genre")

                # Enhanced information
                if game.get('similarity_score'):
                    similarity = game['similarity_score']
                    st.progress(similarity, text=f"Semantic Match: {similarity:.1%}")

                # Rating and platforms
                rating_col, meta_col = st.columns(2)
                with rating_col:
                    if game.get("rating") and game["rating"] != "N/A":
                        rating = game["rating"]
                        rating_top = game.get("rating_top", 5)
                        st.metric("⭐ Rating", f"{rating}/{rating_top}")

                with meta_col:
                    if game.get("metacritic"):
                        metacritic = game["metacritic"]
                        color = "green" if metacritic >= 75 else "orange" if metacritic >= 50 else "red"
                        st.markdown(f"**Metacritic:** <span style='color: {color}'>{metacritic}</span>",
                                    unsafe_allow_html=True)

                # Platform badges
                if game.get("platforms"):
                    platform_text = " • ".join(game["platforms"][:3])
                    st.caption(f"🖥️ {platform_text}")

                # Detailed information with VectorDB enhancements
                with st.expander("📋 Enhanced Details"):
                    if game.get('released') and game['released'] != "N/A":
                        st.write(f"**📅 Released:** {game['released']}")

                    if game.get('genres'):
                        genres_str = ", ".join(game['genres'])
                        st.write(f"🎭 **Genres:** {genres_str}")

                    # Show source information
                    source = game.get('source', 'rawg')
                    if source.startswith('semantic'):
                        st.write("**🔍 Source:** Semantic Search (VectorDB)")
                    elif source == 'favorite_genre_semantic':
                        st.write("**🔍 Source:** Favorite Genre Semantic Search")
                    else:
                        st.write("**🔍 Source:** RAWG API")

                    if game.get('similarity_score'):
                        st.write(f"**🎯 Semantic Similarity:** {game['similarity_score']:.1%}")

                    st.write(f"📖 **Description:** {game.get('description', 'Click for more details')}")


# ---------------------------------------------------------------------
# 🧩 VECTORDB MANAGEMENT INTERFACE
# ---------------------------------------------------------------------
@require_auth()
def show_vectordb_management():
    """VectorDB management interface"""
    st.subheader("🧠 VectorDB Management")

    vectordb = initialize_vectordb()
    if not vectordb:
        st.error("VectorDB not available")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Initialize VectorDB"):
            with st.spinner("Initializing VectorDB..."):
                vectordb = initialize_vectordb()
                if vectordb:
                    st.success("VectorDB initialized!")
                else:
                    st.error("Failed to initialize VectorDB")

    with col2:
        if st.button("📥 Populate from RAWG"):
            with st.spinner("Populating VectorDB with popular games..."):
                if populate_vectordb_from_rawg():
                    st.success("VectorDB populated successfully!")
                else:
                    st.error("Failed to populate VectorDB")

    with col3:
        if st.button("📊 Show Stats"):
            stats = vectordb.get_collection_stats()
            st.metric("Games in VectorDB", stats['total_games'])

    # Semantic search interface
    st.markdown("---")
    st.subheader("🔍 Semantic Search")

    search_query = st.text_input("Search games semantically:", placeholder="Find games similar to...")

    if search_query:
        with st.spinner("Searching semantically..."):
            results = vectordb.semantic_search(query=search_query, n_results=5)

            if results:
                st.success(f"Found {len(results)} semantic matches:")
                for result in results:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{result['name']}**")
                            st.write(f"Genres: {result.get('genres', 'N/A')}")
                            st.write(f"Platforms: {result.get('platforms', 'N/A')}")
                        with col2:
                            distance = result.get('distance', 0)
                            similarity = 1 - distance
                            st.metric("Similarity", f"{similarity:.1%}")
            else:
                st.warning("No semantic matches found.")


# ---------------------------------------------------------------------
# 🧩 GAME MANAGEMENT FOR STANDARD USERS (SQLite) - ENHANCED
# ---------------------------------------------------------------------
@require_auth()
def show_personal_library():
    """Interface for standard users to manage their personal SQLite library"""
    if is_main_user():
        st.warning(
            "🎮 As the main user, you have read-only access to the PostgreSQL database. Use external tools to manage it.")
        return

    st.subheader("📚 My Personal Game Library")

    # ENHANCED: Added VectorDB Search tab
    tab1, tab2, tab3, tab4 = st.tabs(["Add Game", "View My Library", "Enhanced Recommendations", "VectorDB Search"])

    with tab1:
        st.markdown("### Add New Game to My Library")

        with st.form("add_game_form"):
            col1, col2 = st.columns(2)

            with col1:
                game_name = st.text_input("Game Name *", placeholder="The Legend of Zelda: Breath of the Wild")
                platform = st.selectbox("Platform *",
                                        ["PC", "PlayStation", "Xbox", "Nintendo Switch", "Mobile", "Mac", "Other"])
                status = st.selectbox("Status *", ["Completed", "Playing", "Backlog", "Dropped", "Plan to Play"])

            with col2:
                # Genre selection with improved options - Use English genres for standard users
                genre_options = sorted(list(ENGLISH_TO_RAWG_MAPPING.keys()))
                selected_genre = st.selectbox("Main Genre *", genre_options,
                                              index=genre_options.index('action') if 'action' in genre_options else 0)
                custom_genre = st.text_input("Or enter custom genre", placeholder="Leave empty to use selected genre")

            notes = st.text_area("Additional Notes", placeholder="Any additional information about this game...")

            if st.form_submit_button("Add Game to My Library"):
                if not game_name or not platform or not status:
                    st.error("Please fill in all required fields (*)")
                else:
                    # Use custom genre if provided, otherwise use selected genre
                    final_genre = custom_genre.strip() if custom_genre.strip() else selected_genre

                    if add_game_to_sqlite(st.session_state.username, game_name, platform, status, final_genre, notes):
                        st.success(f"✅ '{game_name}' added to your personal library!")
                    else:
                        st.error("❌ Failed to add game to your library.")

    with tab2:
        st.markdown("### My Game Collection")

        user_games = get_user_games(st.session_state.username)

        if user_games:
            # Convert to DataFrame for better display
            games_df = pd.DataFrame(user_games, columns=['ID', 'Name', 'Platform', 'Status', 'Genre', 'Notes'])

            # Status counts
            status_counts = games_df['Status'].value_counts()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Games", len(games_df))
            with col2:
                st.metric("Playing", status_counts.get('Playing', 0))
            with col3:
                st.metric("Completed", status_counts.get('Completed', 0))
            with col4:
                st.metric("Backlog", status_counts.get('Backlog', 0))

            # Search and filter
            search_term = st.text_input("Search my games by name:")
            if search_term:
                filtered_games = games_df[games_df['Name'].str.contains(search_term, case=False, na=False)]
            else:
                filtered_games = games_df

            # Status filter
            status_filter = st.multiselect("Filter by status:", options=games_df['Status'].unique(), default=[])
            if status_filter:
                filtered_games = filtered_games[filtered_games['Status'].isin(status_filter)]

            # Display games with delete option
            for index, game in filtered_games.iterrows():
                with st.container():
                    st.markdown("---")
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.write(f"**{game['Name']}**")
                        st.write(f"Platform: {game['Platform']} | Status: {game['Status']} | Genre: {game['Genre']}")
                        if game['Notes']:
                            st.write(f"Notes: {game['Notes']}")

                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{game['ID']}"):
                            if delete_user_game(st.session_state.username, game['ID']):
                                st.success(f"Deleted '{game['Name']}'")
                                st.rerun()
                            else:
                                st.error("Failed to delete game")

            # Export option
            if st.button("Export My Library to CSV"):
                csv = filtered_games.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="my_personal_game_library.csv",
                    mime="text/csv"
                )
        else:
            st.info("Your personal game library is empty. Add some games to get started!")

    with tab3:
        show_enhanced_standard_user_recommendations()

    with tab4:
        show_vectordb_management()


# ---------------------------------------------------------------------
# 🧩 CORE SYSTEM FUNCTIONS (Read-only PostgreSQL) - ALL YOUR EXISTING FUNCTIONS
# ---------------------------------------------------------------------
def create_rag_sql_chain():
    """Create RAG SQL chain with English prompts - READ ONLY - FIXED"""
    try:
        db_uri = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        db = SQLDatabase.from_uri(db_uri)

        execute_tool = QuerySQLDataBaseTool(db=db)
        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2)

        sql_prompt = ChatPromptTemplate.from_template("""
You are an expert SQL generator for PostgreSQL. Generate valid SQL queries for this Spanish database.

IMPORTANT: This is a READ-ONLY database. You can only use SELECT queries.
Never generate INSERT, UPDATE, DELETE, or any other modifying queries.

DATABASE SCHEMA:
- Table: played_games
- Columns: nombre (game name), genero_principal (main genre)
- Never include games where genero_principal = 'Eroge'

User question: {question}
Database schema: {schema}

Generate SQL that queries the Spanish column names but understands English questions.
Return only the SQL query without explanations.
""")

        sql_chain = (
                {"question": RunnablePassthrough(), "schema": lambda _: db.get_table_info()}
                | sql_prompt
                | llm
                | StrOutputParser()
        )

        summary_prompt = ChatPromptTemplate.from_template("""
You are a game specialist assistant. Analyze the SQL results and respond helpfully.

SQL RESULTS: {result}
ORIGINAL QUESTION: {question}

Respond in English with a clear, helpful answer. If it's a list of games, show them all.
If it's an analysis question, provide insights.
""")

        summary_chain = (
                {"result": RunnablePassthrough(), "question": RunnablePassthrough()}
                | summary_prompt
                | llm
                | StrOutputParser()
        )

        return sql_chain, execute_tool, summary_chain, None
    except Exception as e:
        return None, None, None, str(e)


def extract_game_titles(sql_result):
    """Extract game titles from SQL result - FIXED version"""
    game_titles = []

    # Parse the result first
    parsed_result = parse_sql_result(sql_result)

    st.write(f"🔍 Debug - Parsed SQL result type: {type(parsed_result)}")
    st.write(f"🔍 Debug - Parsed SQL result: {parsed_result}")

    if isinstance(parsed_result, list):
        for item in parsed_result:
            st.write(f"🔍 Debug - Item: {item}, type: {type(item)}")
            if isinstance(item, (list, tuple)) and len(item) > 0:
                title = item[0]
                if title and str(title).strip() and str(title).strip() != "None":
                    clean_title = str(title).strip()
                    game_titles.append(clean_title)
                    st.write(f"🔍 Debug - Added title: {clean_title}")
            elif isinstance(item, str):
                # Handle string items
                clean_item = item.strip()
                if clean_item and clean_item != "None":
                    game_titles.append(clean_item)
                    st.write(f"🔍 Debug - Added string item: {clean_item}")

    elif isinstance(parsed_result, str):
        try:
            # Try to parse string as Python list
            nested_parsed = ast.literal_eval(parsed_result)
            if isinstance(nested_parsed, list):
                for item in nested_parsed:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        title = item[0]
                        if title and str(title).strip():
                            game_titles.append(str(title).strip())
        except (SyntaxError, ValueError):
            # Fallback: regex extraction
            pattern = r"'([^']*)'"
            matches = re.findall(pattern, parsed_result)
            for match in matches:
                if (match and len(match) > 2 and len(match) < 100 and
                        match.lower() not in ['eroge', 'none', 'null', '']):
                    game_titles.append(match)

    # Remove duplicates and empty values
    game_titles = [title for title in game_titles if title and title != "None"]
    unique_titles = list(dict.fromkeys(game_titles))

    st.write(f"🔍 Debug - Final extracted titles: {unique_titles}")
    return unique_titles


def get_genres_from_database(game_titles, sql_chain, execute_tool):
    """Get EXACT genres from PostgreSQL DB for each game"""
    game_genres = {}

    for title in game_titles[:4]:
        try:
            sql_query = f"""
            SELECT nombre, genero_principal 
            FROM played_games 
            WHERE nombre ILIKE '%{title}%' 
            LIMIT 1;
            """

            result = execute_tool.invoke({"query": sql_query})
            parsed_result = parse_sql_result(result)
            genre = extract_genre_from_result(parsed_result, title)
            if genre:
                game_genres[title] = genre
            else:
                game_genres[title] = 'default'

        except Exception as e:
            print(f"Error getting genre for {title}: {e}")
            game_genres[title] = 'default'

    return game_genres


def extract_genre_from_result(result, title):
    """Extract specific genre from SQL result"""
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                if str(item[0]).lower() == title.lower():
                    return item[1]
    elif isinstance(result, str):
        for genre in GENRE_MAPPING.keys():
            if genre in result and genre != 'default':
                return genre
    return 'default'


def get_detailed_game_info(game):
    """Get detailed game information"""
    game_info = {
        "name": game.get("name", "N/A"),
        "released": game.get("released", "N/A"),
        "rating": game.get("rating", "N/A"),
        "rating_top": game.get("rating_top", 5),
        "metacritic": game.get("metacritic"),
        "playtime": game.get("playtime", "N/A"),
        "genres": [g["name"] for g in game.get("genres", [])],
        "platforms": [p["platform"]["name"] for p in game.get("platforms", [])][:3],
        "stores": [s["store"]["name"] for s in game.get("stores", [])][:2],
        "background_image": game.get("background_image"),
        "description": "Click for more details",
        "id": game.get("id")
    }

    try:
        game_id = game.get("id")
        if game_id:
            headers = {"User-Agent": "VideoGameRAG/1.0", "Accept": "application/json"}
            detail_response = requests.get(
                f"https://api.rawg.io/api/games/{game_id}",
                params={"key": RAWG_API_KEY},
                headers=headers,
                timeout=5
            )
            if detail_response.status_code == 200:
                detail_data = detail_response.json()
                description = detail_data.get("description_raw", "")
                if description and len(description) > 50:
                    if len(description) > 200:
                        game_info["description"] = description[:200] + "..."
                    else:
                        game_info["description"] = description
    except:
        pass

    return game_info


def remove_duplicate_recommendations(recommendations):
    """Remove duplicate recommendations"""
    seen = set()
    unique = []
    for rec in recommendations:
        if rec["name"] not in seen:
            seen.add(rec["name"])
            unique.append(rec)
    return unique


def get_platform_ids(preferred_platforms):
    """Convert platform names to RAWG API platform IDs"""
    platform_ids = []
    for platform in preferred_platforms:
        if platform in PLATFORM_MAPPING:
            platform_ids.extend(PLATFORM_MAPPING[platform])
    return list(set(platform_ids))


def filter_games_by_platforms(games, preferred_platforms):
    """Filter games to only include preferred platforms"""
    if not preferred_platforms:
        return games

    filtered_games = []
    for game in games:
        game_platforms = [p.lower() for p in game.get("platforms", [])]
        # Check if any preferred platform is in the game's platforms
        if any(any(pref.lower() in platform for platform in game_platforms) for pref in preferred_platforms):
            filtered_games.append(game)

    return filtered_games


def get_varied_recommendations(game_titles, sql_chain, execute_tool, user_prefs):
    """Get varied recommendations based on user preferences"""
    all_recommendations = []
    headers = {"User-Agent": "VideoGameRAG/1.0", "Accept": "application/json"}

    # Get user preferences
    rating_threshold = user_prefs.get('rating_threshold', 3.5)
    favorite_genres = user_prefs.get('favorite_genres', [])
    preferred_platforms = user_prefs.get('preferred_platforms', [])
    variety_level = user_prefs.get('variety_level', 3)

    # 1. Get genres from PostgreSQL database
    game_genres = get_genres_from_database(game_titles, sql_chain, execute_tool)

    if game_genres:
        st.info(f"🎯 **Genres found in library:** {list(set(game_genres.values()))}")

    # 2. Strategy 1: Direct genre-based recommendations (similar games)
    similar_recommendations = []
    for title, my_genre in game_genres.items():
        rawg_genres = GENRE_MAPPING.get(my_genre, GENRE_MAPPING['default'])

        if rawg_genres:
            # Use variety level to determine how many different searches to perform
            search_count = min(variety_level, 3)

            for i in range(search_count):
                # Vary the ordering to get different results
                orderings = ["-rating", "-metacritic", "-added", "-released"]
                selected_ordering = orderings[i % len(orderings)]

                similar_params = {
                    "key": RAWG_API_KEY,
                    "genres": ",".join(rawg_genres[:2]),
                    "page_size": 6,
                    "ordering": selected_ordering,
                    "page": random.randint(1, 3)  # Random page for variety
                }

                # Add platform filter if user has preferences
                platform_ids = get_platform_ids(preferred_platforms)
                if platform_ids:
                    similar_params["platforms"] = ",".join(map(str, platform_ids[:3]))

                try:
                    similar_response = requests.get(
                        "https://api.rawg.io/api/games",
                        params=similar_params,
                        headers=headers,
                        timeout=10
                    )

                    if similar_response.status_code == 200:
                        similar_games = similar_response.json().get("results", [])

                        for game in similar_games:
                            game_name = game.get("name")
                            game_rating = game.get("rating", 0)

                            # Apply user preferences filters
                            if game_rating < rating_threshold:
                                continue

                            if (game_name.lower() != title.lower() and
                                    not any(rec["name"] == game_name for rec in similar_recommendations)):
                                game_info = get_detailed_game_info(game)
                                similar_recommendations.append(game_info)

                except Exception as e:
                    continue

    # 3. Strategy 2: Favorite genre recommendations (if user has favorites)
    favorite_recommendations = []
    if favorite_genres and variety_level >= 2:
        for fav_genre in favorite_genres[:2]:  # Use top 2 favorite genres
            # Convert favorite genre to Spanish for PostgreSQL mapping
            spanish_genre = ENGLISH_TO_SPANISH_GENRES.get(fav_genre.lower(), fav_genre)
            rawg_genres = GENRE_MAPPING.get(spanish_genre, [])

            if rawg_genres:
                fav_params = {
                    "key": RAWG_API_KEY,
                    "genres": ",".join(rawg_genres),
                    "page_size": 4,
                    "ordering": "-rating,-metacritic",
                    "page": random.randint(1, 2)
                }

                # Add platform filter if user has preferences
                platform_ids = get_platform_ids(preferred_platforms)
                if platform_ids:
                    fav_params["platforms"] = ",".join(map(str, platform_ids[:3]))

                try:
                    fav_response = requests.get(
                        "https://api.rawg.io/api/games",
                        params=fav_params,
                        headers=headers,
                        timeout=10
                    )

                    if fav_response.status_code == 200:
                        fav_games = fav_response.json().get("results", [])

                        for game in fav_games:
                            game_name = game.get("name")
                            game_rating = game.get("rating", 0)

                            if game_rating >= rating_threshold:
                                game_info = get_detailed_game_info(game)
                                favorite_recommendations.append(game_info)

                except Exception as e:
                    continue

    # 4. Strategy 3: Popular games from preferred platforms (for high variety)
    platform_recommendations = []
    if preferred_platforms and variety_level >= 4:
        platform_ids = get_platform_ids(preferred_platforms)
        if platform_ids:
            platform_params = {
                "key": RAWG_API_KEY,
                "platforms": ",".join(map(str, platform_ids[:3])),
                "page_size": 4,
                "ordering": "-rating",
                "page": random.randint(1, 2)
            }

            try:
                platform_response = requests.get(
                    "https://api.rawg.io/api/games",
                    params=platform_params,
                    headers=headers,
                    timeout=10
                )

                if platform_response.status_code == 200:
                    platform_games = platform_response.json().get("results", [])

                    for game in platform_games:
                        game_rating = game.get("rating", 0)
                        if game_rating >= rating_threshold:
                            game_info = get_detailed_game_info(game)
                            platform_recommendations.append(game_info)

            except Exception as e:
                pass

    # 5. Combine all strategies based on variety level
    all_recommendations = similar_recommendations

    if variety_level >= 2:
        all_recommendations.extend(favorite_recommendations)

    if variety_level >= 4:
        all_recommendations.extend(platform_recommendations)

    # Remove duplicates
    unique_recommendations = remove_duplicate_recommendations(all_recommendations)

    # Apply platform filtering as final step
    if preferred_platforms:
        unique_recommendations = filter_games_by_platforms(unique_recommendations, preferred_platforms)

    # Prioritize favorite genres
    if favorite_genres and unique_recommendations:
        prioritized = []
        others = []

        for rec in unique_recommendations:
            rec_genres = [g.lower() for g in rec.get("genres", [])]
            if any(fav.lower() in ' '.join(rec_genres) for fav in favorite_genres):
                prioritized.append(rec)
            else:
                others.append(rec)

        unique_recommendations = prioritized + others

    # Shuffle for more variety, but keep high-rated games at the front
    if len(unique_recommendations) > 1:
        # Sort by rating first
        unique_recommendations.sort(key=lambda x: (x.get('metacritic', 0) or 0, x.get('rating', 0) or 0), reverse=True)

        # Then shuffle the lower-rated portion based on variety level
        if variety_level >= 3 and len(unique_recommendations) > 4:
            keep_ordered = unique_recommendations[:2]  # Keep top 2 ordered
            to_shuffle = unique_recommendations[2:]  # Shuffle the rest
            random.shuffle(to_shuffle)
            unique_recommendations = keep_ordered + to_shuffle

    return unique_recommendations[:10]  # Return up to 10 recommendations


@require_auth()
def get_genre_specific_recommendations(game_titles, sql_chain, execute_tool):
    """Main recommendation function that uses varied strategies"""
    user_prefs = st.session_state.get('user_preferences', {})
    return get_varied_recommendations(game_titles, sql_chain, execute_tool, user_prefs)


@require_auth()
def display_recommendations(recommendations):
    """Display recommendations attractively"""
    if not recommendations:
        st.warning("No recommendations found for these games.")
        return

    st.subheader("🎮 Personalized Recommendations")

    # Show user preference info
    user_prefs = st.session_state.get('user_preferences', {})
    if user_prefs.get('favorite_genres'):
        st.caption(f"✨ Prioritizing your favorite genres: {', '.join(user_prefs['favorite_genres'])}")
    if user_prefs.get('preferred_platforms'):
        st.caption(f"🎯 Preferred platforms: {', '.join(user_prefs['preferred_platforms'])}")
    if user_prefs.get('rating_threshold', 3.5) > 3.5:
        st.caption(f"⭐ Showing games rated {user_prefs['rating_threshold']}+")
    if user_prefs.get('variety_level', 3) > 3:
        st.caption(f"🔄 High variety mode: Discovering diverse games")

    # Show recommendations in columns
    cols = st.columns(2)

    for idx, game in enumerate(recommendations):
        with cols[idx % 2]:
            with st.container():
                st.markdown("---")

                # Header with image and title
                col1, col2 = st.columns([1, 2])

                with col1:
                    if game["background_image"]:
                        st.image(game["background_image"], width=80)
                    else:
                        st.image("🎮", width=80)

                with col2:
                    st.markdown(f"### {game['name']}")

                    # Rating and Metacritic
                    rating_col, meta_col = st.columns(2)
                    with rating_col:
                        if game["rating"] != "N/A":
                            rating = game["rating"]
                            rating_top = game["rating_top"]
                            st.metric("⭐ Rating", f"{rating}/{rating_top}")

                    with meta_col:
                        if game["metacritic"]:
                            metacritic = game["metacritic"]
                            color = "green" if metacritic >= 75 else "orange" if metacritic >= 50 else "red"
                            st.markdown(f"**Metacritic:** <span style='color: {color}'>{metacritic}</span>",
                                        unsafe_allow_html=True)

                # Platform badges
                if game["platforms"]:
                    platform_text = " • ".join(game["platforms"][:3])
                    st.caption(f"🖥️ {platform_text}")

                # Detailed information
                with st.expander("📋 View details"):
                    if game["released"] != "N/A":
                        st.write(f"**📅 Released:** {game['released']}")

                    if game["genres"]:
                        genres_str = ", ".join(game["genres"])
                        st.write(f"🎭 **Genres:** {genres_str}")

                    if game["playtime"] != "N/A":
                        st.write(f"⏱️ **Average playtime:** {game['playtime']} hours")

                    st.write(f"📖 **Description:** {game['description']}")

                    if game["stores"]:
                        stores_str = ", ".join(game["stores"])
                        st.write(f"🛒 **Available on:** {stores_str}")


def get_standard_user_recommendations(user_games):
    """Get recommendations for standard users based on their SQLite games - FIXED"""
    all_recommendations = []
    headers = {"User-Agent": "VideoGameRAG/1.0", "Accept": "application/json"}

    # Get user preferences
    user_prefs = st.session_state.get('user_preferences', {})
    rating_threshold = user_prefs.get('rating_threshold', 3.5)
    favorite_genres = user_prefs.get('favorite_genres', [])
    preferred_platforms = user_prefs.get('preferred_platforms', [])
    variety_level = user_prefs.get('variety_level', 3)

    # Extract genres from user's selected games
    game_genres = {}
    for game_name, genre in user_games:
        game_genres[game_name] = genre.lower()  # Convert to lowercase for consistent matching

    if game_genres:
        unique_genres = list(set(game_genres.values()))
        st.info(f"🎯 **Genres from your selected games:** {', '.join(unique_genres)}")

    # Strategy 1: Direct genre-based recommendations (similar games)
    similar_recommendations = []
    for title, user_genre in game_genres.items():
        # Map user's genre to RAWG genres - USE NEW DIRECT MAPPING
        rawg_genres = []

        # Direct mapping from user's genre to RAWG genres
        if user_genre in ENGLISH_TO_RAWG_MAPPING:
            rawg_genres = ENGLISH_TO_RAWG_MAPPING[user_genre]
        else:
            # Try partial matching
            for genre_key, rawg_list in ENGLISH_TO_RAWG_MAPPING.items():
                if genre_key in user_genre or user_genre in genre_key:
                    rawg_genres = rawg_list
                    break

        # Fallback to default if no mapping found
        if not rawg_genres:
            rawg_genres = ENGLISH_TO_RAWG_MAPPING['default']
            st.warning(f"⚠️ Couldn't find exact genre mapping for '{user_genre}'. Using general recommendations.")

        st.write(f"🔍 **Genre Mapping:** '{user_genre}' → RAWG genres: {rawg_genres}")

        if rawg_genres:
            # Use variety level to determine how many different searches to perform
            search_count = min(variety_level, 3)

            for i in range(search_count):
                # Vary the ordering to get different results
                orderings = ["-rating", "-metacritic", "-added", "-released"]
                selected_ordering = orderings[i % len(orderings)]

                similar_params = {
                    "key": RAWG_API_KEY,
                    "genres": ",".join(rawg_genres[:2]),  # Use first 2 genres max
                    "page_size": 8,  # Increased to get more results
                    "ordering": selected_ordering,
                    "page": random.randint(1, 2)  # Use first 2 pages for more variety
                }

                # Add platform filter if user has preferences
                platform_ids = get_platform_ids(preferred_platforms)
                if platform_ids:
                    similar_params["platforms"] = ",".join(map(str, platform_ids[:3]))

                try:
                    similar_response = requests.get(
                        "https://api.rawg.io/api/games",
                        params=similar_params,
                        headers=headers,
                        timeout=10
                    )

                    if similar_response.status_code == 200:
                        similar_games = similar_response.json().get("results", [])

                        for game in similar_games:
                            game_name = game.get("name")
                            game_rating = game.get("rating", 0)

                            # Apply user preferences filters
                            if game_rating < rating_threshold:
                                continue

                            # Skip if it's the same game or already in recommendations
                            if (game_name.lower() != title.lower() and
                                    not any(rec["name"] == game_name for rec in similar_recommendations)):

                                # Additional genre matching check
                                game_rawg_genres = [g["name"].lower() for g in game.get("genres", [])]
                                has_matching_genre = any(
                                    any(rawg_genre in game_genre for game_genre in game_rawg_genres)
                                    for rawg_genre in rawg_genres
                                )

                                if has_matching_genre:
                                    game_info = get_detailed_game_info(game)
                                    similar_recommendations.append(game_info)

                except Exception as e:
                    continue

    # Strategy 2: Favorite genre recommendations (if user has favorites)
    favorite_recommendations = []
    if favorite_genres and variety_level >= 2:
        st.info("🎭 **Also including recommendations from your favorite genres**")
        for fav_genre in favorite_genres[:2]:  # Use top 2 favorite genres
            if fav_genre.lower() in ENGLISH_TO_RAWG_MAPPING:
                rawg_genres = ENGLISH_TO_RAWG_MAPPING[fav_genre.lower()]

                if rawg_genres:
                    fav_params = {
                        "key": RAWG_API_KEY,
                        "genres": ",".join(rawg_genres),
                        "page_size": 4,
                        "ordering": "-rating,-metacritic",
                        "page": 1
                    }

                    # Add platform filter if user has preferences
                    platform_ids = get_platform_ids(preferred_platforms)
                    if platform_ids:
                        fav_params["platforms"] = ",".join(map(str, platform_ids[:3]))

                    try:
                        fav_response = requests.get(
                            "https://api.rawg.io/api/games",
                            params=fav_params,
                            headers=headers,
                            timeout=10
                        )

                        if fav_response.status_code == 200:
                            fav_games = fav_response.json().get("results", [])

                            for game in fav_games:
                                game_name = game.get("name")
                                game_rating = game.get("rating", 0)

                                if game_rating >= rating_threshold:
                                    game_info = get_detailed_game_info(game)
                                    # Check if not already in similar recommendations
                                    if not any(rec["name"] == game_name for rec in similar_recommendations):
                                        favorite_recommendations.append(game_info)

                    except Exception as e:
                        continue

    # Strategy 3: Popular games from preferred platforms (for high variety)
    platform_recommendations = []
    if preferred_platforms and variety_level >= 4:
        platform_ids = get_platform_ids(preferred_platforms)
        if platform_ids:
            platform_params = {
                "key": RAWG_API_KEY,
                "platforms": ",".join(map(str, platform_ids[:3])),
                "page_size": 4,
                "ordering": "-rating",
                "page": 1
            }

            try:
                platform_response = requests.get(
                    "https://api.rawg.io/api/games",
                    params=platform_params,
                    headers=headers,
                    timeout=10
                )

                if platform_response.status_code == 200:
                    platform_games = platform_response.json().get("results", [])

                    for game in platform_games:
                        game_rating = game.get("rating", 0)
                        if game_rating >= rating_threshold:
                            game_info = get_detailed_game_info(game)
                            # Check if not already in other recommendations
                            if (not any(rec["name"] == game_info["name"] for rec in similar_recommendations) and
                                    not any(rec["name"] == game_info["name"] for rec in favorite_recommendations)):
                                platform_recommendations.append(game_info)

            except Exception as e:
                pass

    # Combine all strategies based on variety level
    all_recommendations = similar_recommendations

    if variety_level >= 2:
        all_recommendations.extend(favorite_recommendations)

    if variety_level >= 4:
        all_recommendations.extend(platform_recommendations)

    # Remove duplicates
    unique_recommendations = remove_duplicate_recommendations(all_recommendations)

    # Apply platform filtering as final step
    if preferred_platforms:
        unique_recommendations = filter_games_by_platforms(unique_recommendations, preferred_platforms)

    # Prioritize games that match the original genres
    if game_genres and unique_recommendations:
        target_genres = list(set(game_genres.values()))
        prioritized = []
        others = []

        for rec in unique_recommendations:
            rec_genres = [g.lower() for g in rec.get("genres", [])]
            # Check if any of the recommendation's genres match the target genres
            genre_match = any(
                any(target_genre in rec_genre for rec_genre in rec_genres)
                for target_genre in target_genres
            )

            if genre_match:
                prioritized.append(rec)
            else:
                others.append(rec)

        unique_recommendations = prioritized + others

    # Also prioritize favorite genres
    if favorite_genres and unique_recommendations:
        prioritized_fav = []
        others_fav = []

        for rec in unique_recommendations:
            rec_genres = [g.lower() for g in rec.get("genres", [])]
            fav_match = any(
                any(fav_genre.lower() in rec_genre for rec_genre in rec_genres)
                for fav_genre in favorite_genres
            )

            if fav_match:
                prioritized_fav.append(rec)
            else:
                others_fav.append(rec)

        unique_recommendations = prioritized_fav + others_fav

    # Sort by relevance (rating and metacritic)
    if len(unique_recommendations) > 1:
        unique_recommendations.sort(
            key=lambda x: (x.get('metacritic', 0) or 0, x.get('rating', 0) or 0),
            reverse=True
        )

    # Show debug info about found recommendations
    if unique_recommendations:
        found_genres = set()
        for rec in unique_recommendations:
            found_genres.update([g.lower() for g in rec.get("genres", [])])
        st.success(
            f"✅ Found {len(unique_recommendations)} recommendations across genres: {', '.join(sorted(found_genres)[:5])}")

    return unique_recommendations[:12]  # Return up to 12 recommendations


@require_auth()
def show_standard_user_recommendations():
    """Show recommendations for standard users based on their SQLite library"""
    st.markdown("### 🎯 Get Personalized Recommendations")

    # Check if user has games in their library
    user_game_count = get_user_games_count(st.session_state.username)

    if user_game_count == 0:
        st.warning("📚 You need to add at least one game to your library to get recommendations!")
        st.info("💡 Go to the 'Add Game' tab to add some games you've played or want to play.")
        return

    st.success(f"🎮 Found {user_game_count} games in your library!")

    # Let user choose which games to base recommendations on
    st.markdown("#### Select games to base recommendations on:")

    # Get user's games by status
    completed_games = get_user_games_by_status(st.session_state.username, "Completed")
    playing_games = get_user_games_by_status(st.session_state.username, "Playing")

    all_user_games = completed_games + playing_games

    if not all_user_games:
        st.warning("Add some 'Completed' or 'Playing' games to get better recommendations!")
        all_user_games = get_user_games(st.session_state.username)
        if all_user_games:
            all_user_games = [(game[1], game[4]) for game in all_user_games]  # (name, genre)

    if all_user_games:
        # Let user select games for recommendations
        game_options = [f"{game[0]} ({game[1]})" for game in all_user_games]
        selected_game_indices = st.multiselect(
            "Choose games you like (select 1-3 for best results):",
            options=range(len(game_options)),
            format_func=lambda x: game_options[x],
            max_selections=3
        )

        if selected_game_indices:
            selected_games = [all_user_games[i] for i in selected_game_indices]
            game_titles = [game[0] for game in selected_games]
            game_genres = [game[1] for game in selected_games]

            st.info(f"🎯 Getting recommendations based on: {', '.join(game_titles)}")
            st.info(f"🎭 Selected genres: {', '.join(set(game_genres))}")

            if st.button("🎮 Get Recommendations", type="primary"):
                with st.spinner("Finding personalized recommendations..."):
                    recommendations = get_standard_user_recommendations(selected_games)

                if recommendations:
                    display_recommendations(recommendations)
                else:
                    st.warning("No recommendations found. Try selecting different games or check your preferences.")
        else:
            st.info("👆 Select 1-3 games from your library to get personalized recommendations")
    else:
        st.warning("No games found in your library. Add some games first!")


@require_auth()
def show_about_page():
    """About page"""
    st.subheader("About Game Assistant")

    st.markdown("""
    ### 🎮 Intelligent Game Assistant

    This application helps you discover and explore games using natural language queries
    and AI-powered recommendations.

    **User Types:**
    - **🎮 Main User**: Read-only access to PostgreSQL game library + recommendations
    - **👤 Standard Users**: Personal SQLite libraries + game discovery + recommendations

    **Key Features:**
    - 🔍 **Natural Language Queries**: Ask about games in plain English
    - 🎯 **Personalized Recommendations**: Get suggestions based on your preferences and library
    - 📚 **Personal Game Libraries**: Standard users can manage their own collections
    - 🛡️ **Safe Content Filtering**: Automatic exclusion of inappropriate content
    - 👤 **User Profiles**: Save your preferences and game tastes

    **Database Access:**
    - **PostgreSQL**: Read-only access for main user (managed externally)
    - **SQLite**: Personal databases for standard users (managed in-app)

    **Tech Stack:**
    - **Frontend**: Streamlit
    - **AI/LLM**: Google Gemini
    - **Database**: PostgreSQL (read-only) + SQLite (personal libraries)
    - **Game Data**: RAWG.io API
    - **Authentication**: Custom SQLite-based system

    *Built with ❤️ for gamers everywhere*
    """)


# ---------------------------------------------------------------------
# 🎯 MAIN APPLICATION INTERFACE
# ---------------------------------------------------------------------
def main():
    """Main application interface"""

    # Check authentication
    if not check_authentication():
        return

    # User is authenticated, show main app
    show_main_interface()


@require_auth()
def show_main_interface():
    """Show the main application interface for authenticated users"""

    # User header with profile and logout
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.title("🎮 Intelligent Game Assistant")
        if is_main_user():
            st.caption("🎮 PostgreSQL Access (Read-Only) - Your existing game library")
        else:
            st.caption("👤 Personal SQLite Library - Your personal game collection + Recommendations")

    with col2:
        st.write(f"👤 **Welcome, {st.session_state.username}!**")

    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            auth_system = initialize_auth()
            if auth_system.logout_user(st.session_state.user_session):
                for key in ['user_session', 'username', 'user_preferences']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    st.markdown("---")

    # Navigation options based on user type
    if is_main_user():
        # Main user gets PostgreSQL query access
        nav_options = ["Query Library", "User Profile", "About"]
    else:
        # Standard users get personal library management and recommendations
        nav_options = ["My Library", "User Profile", "About"]

    page = st.sidebar.selectbox("Navigation", nav_options)

    if page == "Query Library" and is_main_user():
        show_postgresql_query()
    elif page == "My Library" and not is_main_user():
        show_personal_library()
    elif page == "User Profile":
        show_user_profile()
    elif page == "About":
        show_about_page()


@require_auth()
def show_postgresql_query():
    """PostgreSQL query interface - Only for main user, read-only"""
    if not is_main_user():
        st.warning("🔒 This feature is only available for the main user.")
        return

    # Add debug information at the top
    with st.expander("🔧 Database Status & Debug Info", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧪 Test Database Connection"):
                has_games = has_postgresql_games()
                if has_games:
                    st.success("✅ PostgreSQL database has games and is accessible!")
                else:
                    st.error("❌ No games found or database connection issue")

        with col2:
            if st.button("📊 Show Detailed Debug Info"):
                debug_postgresql_connection()

    if not has_postgresql_games():
        st.warning("📚 The PostgreSQL game library is empty or unavailable.")
        st.info("💡 Use the debug tools above to diagnose the issue.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Query Game Library (Read-Only)")

        with st.expander("📋 Example Questions"):
            st.markdown("""
            - **"What fighting games are in the library?"**
            - **"Show me all RPG games"**  
            - **"What platformer games are available?"**
            - **"List games with action-adventure genre"**
            - **"What strategy games are in the collection?"**
            - **"Show me games from specific platforms"**
            """)

        user_question = st.text_input(
            "Ask about the game library:",
            placeholder="Example: What fighting games are available?",
            key="library_question"
        )

    with col2:
        st.subheader("🎯 Settings")
        show_sql = st.checkbox("Show SQL query", value=False)
        get_recommendations = st.checkbox("Get recommendations", value=True)

        # Show user preference summary
        user_prefs = st.session_state.get('user_preferences', {})
        if user_prefs.get('favorite_genres'):
            st.caption(f"❤️ Favorite genres: {', '.join(user_prefs['favorite_genres'][:2])}")
        if user_prefs.get('preferred_platforms'):
            st.caption(f"🎯 Platforms: {', '.join(user_prefs['preferred_platforms'])}")
        if user_prefs.get('rating_threshold'):
            st.caption(f"⭐ Min rating: {user_prefs['rating_threshold']}")
        if user_prefs.get('variety_level'):
            st.caption(f"🔄 Variety: {user_prefs['variety_level']}/5")

    # Process library search
    if user_question:
        with st.spinner("🔍 Querying library..."):
            try:
                sql_chain, execute_tool, summary_chain, error = create_rag_sql_chain()

                if error:
                    st.error(f"❌ Connection error: {error}")
                    st.stop()

                sql_query = sql_chain.invoke(user_question).strip()
                sql_query = re.sub(r"^```sql|```$", "", sql_query, flags=re.IGNORECASE | re.MULTILINE).strip()
                sql_query = sql_query.replace("```", "").strip()

                # Add safe filter
                if "WHERE" in sql_query.upper():
                    sql_query = sql_query.replace("WHERE", "WHERE genero_principal != 'Eroge' AND")
                else:
                    sql_query += " WHERE genero_principal != 'Eroge'"

                if show_sql:
                    with st.expander("🧩 Generated SQL Query"):
                        st.code(sql_query, language="sql")

                result = execute_tool.invoke({"query": sql_query})
                parsed_result = parse_sql_result(result)
                respuesta = summary_chain.invoke({"result": str(parsed_result), "question": user_question})

                st.subheader("📊 Library Results")
                st.write(respuesta)

                if get_recommendations:
                    game_titles = extract_game_titles(parsed_result)

                    if game_titles:
                        st.info(f"🎯 **Games found:** {game_titles[:5]}")

                        with st.spinner("🎯 Finding personalized recommendations..."):
                            recommendations = get_genre_specific_recommendations(game_titles, sql_chain, execute_tool)

                        display_recommendations(recommendations)
                    else:
                        st.warning("No game titles found for recommendations.")

            except Exception as e:
                st.error(f"❌ Error processing your question: {str(e)}")

    else:
        st.info("👆 Type a question to query the PostgreSQL game library")

        with st.expander("🚀 How It Works"):
            st.markdown("""
            **1. Ask Natural Questions**
            - Type questions about the game library in English
            - The AI understands your intent and generates SQL

            **2. Query PostgreSQL Database**  
            - Read-only access to the existing game library
            - The system automatically bridges the language gap

            **3. Get Smart Results**
            - See library results in English
            - Receive personalized recommendations

            **4. Personalization**
            - Set your favorite genres in User Profile
            - Choose preferred platforms
            - Adjust rating thresholds and variety levels
            - Get tailored suggestions

            **Note**: The PostgreSQL database is read-only. 
            Use external tools to manage the game library.
            """)


def search_rawg_games(query):
    """Search games on RAWG API"""
    headers = {"User-Agent": "VideoGameRAG/1.0", "Accept": "application/json"}

    try:
        response = requests.get(
            "https://api.rawg.io/api/games",
            params={
                "key": RAWG_API_KEY,
                "search": query,
                "page_size": 10,
                "search_precise": True
            },
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            return response.json().get("results", [])
    except Exception as e:
        print(f"RAWG search error: {e}")

    return []


def map_rawg_genre_to_custom(rawg_genres):
    """Map RAWG genres to our custom genre system"""
    if not rawg_genres:
        return "Action"  # Default

    # Try to find a direct match first
    for rawg_genre in rawg_genres:
        rawg_lower = rawg_genre.lower()
        for custom_genre, rawg_mappings in GENRE_MAPPING.items():
            if custom_genre != 'default' and any(mapping in rawg_lower for mapping in rawg_mappings):
                return custom_genre

    # Fallback: use the first RAWG genre
    return rawg_genres[0]


# ---------------------------------------------------------------------
# 🧩 ENHANCED SIDEBAR CONTENT
# ---------------------------------------------------------------------
def setup_sidebar():
    """Setup sidebar content"""
    with st.sidebar:
        st.header("ℹ️ Information")

        if is_main_user():
            st.markdown("""
            **Your Role:** 🎮 Main User
            **Features:**
            - Query PostgreSQL game library (read-only)
            - Get personalized recommendations  
            - Set game preferences
            """)

            # DEBUG OPTION FOR MAIN USER
            st.markdown("---")
            st.header("🔧 Debug Tools")
            if st.button("🔍 Debug PostgreSQL"):
                debug_postgresql_connection()

        else:
            st.markdown("""
            **Your Role:** 👤 Standard User
            **Features:**
            - Manage personal SQLite game library
            - Get enhanced recommendations 🧠
            - Set preferences for recommendations
            - VectorDB semantic search
            """)

        # VectorDB Status
        st.markdown("---")
        st.header("🧠 VectorDB Status")
        vectordb = initialize_vectordb()
        if vectordb:
            try:
                stats = vectordb.get_collection_stats()
                st.success(f"✅ VectorDB Active")
                st.metric("Games in VectorDB", stats['total_games'])
            except:
                st.warning("⚠️ VectorDB Needs Setup")
        else:
            st.warning("⚠️ VectorDB Not Initialized")

        st.header("🎭 Supported Genres")
        with st.expander("View all genres"):
            for genre in sorted(list(ENGLISH_TO_RAWG_MAPPING.keys())):
                st.write(f"• {genre.title()}")

        st.header("🔧 Tools")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Test Connection"):
                if is_main_user():
                    sql_chain, execute_tool, summary_chain, error = create_rag_sql_chain()
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success("✅ PostgreSQL Connected!")
                else:
                    game_count = get_user_games_count(st.session_state.username)
                    st.success(f"✅ Personal Library Ready! ({game_count} games)")

        with col2:
            if st.button("Clear Session"):
                for key in list(st.session_state.keys()):
                    if key not in ['user_session', 'username', 'user_preferences']:
                        del st.session_state[key]
                st.rerun()

        # User stats
        if 'username' in st.session_state:
            st.markdown("---")
            st.header("👤 Your Stats")
            user_prefs = st.session_state.get('user_preferences', {})
            if user_prefs.get('favorite_genres'):
                st.write(f"**Favorite Genres:** {len(user_prefs['favorite_genres'])}")
            if user_prefs.get('preferred_platforms'):
                st.write(f"**Platforms:** {len(user_prefs['preferred_platforms'])}")
            if user_prefs.get('variety_level'):
                st.write(f"**Variety Level:** {user_prefs['variety_level']}/5")

            if not is_main_user():
                game_count = get_user_games_count(st.session_state.username)
                st.write(f"**My Games:** {game_count}")


# ---------------------------------------------------------------------
# 🚀 APPLICATION ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Setup sidebar
    setup_sidebar()

    # Run main application
    main()

    # Enhanced footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by Gemini + PostgreSQL + RAWG.io + SQLite + ChromaDB 🧠 | "
        "Built with Streamlit | "
        "Enhanced with Vector Semantic Search"
        "</div>",
        unsafe_allow_html=True
    )