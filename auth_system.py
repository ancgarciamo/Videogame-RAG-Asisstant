import streamlit as st
import hashlib
import sqlite3
import os
from datetime import datetime, timedelta
import jwt
import re


class AuthenticationSystem:
    def __init__(self):
        self.db_path = "users.db"
        self.secret_key = os.getenv("JWT_SECRET", "game_assistant_secret_key_2024")
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for user management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')

        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')

        # User game preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                username TEXT PRIMARY KEY,
                favorite_genres TEXT,
                preferred_platforms TEXT,
                rating_threshold REAL DEFAULT 3.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')

        conn.commit()
        conn.close()

    def hash_password(self, password):
        """Hash password using SHA-256 with salt"""
        salt = "game_assistant_salt_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def validate_email(self, email):
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def validate_password(self, password):
        """Validate password strength"""
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        return True, "Password is valid"

    def create_user(self, username, email, password):
        """Register new user"""
        try:
            # Validate inputs
            if not username or not email or not password:
                return False, "All fields are required"

            if len(username) < 3:
                return False, "Username must be at least 3 characters long"

            if not self.validate_email(email):
                return False, "Invalid email format"

            is_valid, msg = self.validate_password(password)
            if not is_valid:
                return False, msg

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            password_hash = self.hash_password(password)

            cursor.execute('''
                INSERT INTO users (username, email, password_hash, preferences)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, '{}'))

            # Initialize user preferences with empty values
            cursor.execute('''
                INSERT INTO user_preferences (username, favorite_genres, preferred_platforms, rating_threshold)
                VALUES (?, ?, ?, ?)
            ''', (username, '[]', '[]', 3.5))

            conn.commit()
            conn.close()
            return True, "Account created successfully!"
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                return False, "Username already exists"
            elif "email" in str(e):
                return False, "Email already registered"
            else:
                return False, "Registration failed - user might already exist"
        except Exception as e:
            return False, f"Registration error: {str(e)}"

    def verify_user(self, username, password):
        """Verify user credentials"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT password_hash, is_active FROM users WHERE username = ?
            ''', (username,))

            result = cursor.fetchone()
            conn.close()

            if result and result[0] == self.hash_password(password):
                if not result[1]:  # Check if user is active
                    return False, "Account is deactivated"
                self.update_last_login(username)
                return True, "Login successful"
            return False, "Invalid username or password"
        except Exception as e:
            return False, f"Login error: {str(e)}"

    def update_last_login(self, username):
        """Update user's last login timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE users SET last_login = ? WHERE username = ?
            ''', (datetime.now(), username))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating last login: {e}")

    def create_session(self, username):
        """Create user session"""
        try:
            session_id = hashlib.sha256(f"{username}{datetime.now()}{os.urandom(16)}".encode()).hexdigest()
            expires_at = datetime.now() + timedelta(hours=24)  # 24-hour session

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Clean up expired sessions
            cursor.execute('DELETE FROM user_sessions WHERE expires_at < ?', (datetime.now(),))

            cursor.execute('''
                INSERT INTO user_sessions (session_id, username, expires_at)
                VALUES (?, ?, ?)
            ''', (session_id, username, expires_at))

            conn.commit()
            conn.close()

            return session_id
        except Exception as e:
            print(f"Error creating session: {e}")
            return None

    def validate_session(self, session_id):
        """Validate user session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT username FROM user_sessions 
                WHERE session_id = ? AND expires_at > ? AND 
                username IN (SELECT username FROM users WHERE is_active = TRUE)
            ''', (session_id, datetime.now()))

            result = cursor.fetchone()
            conn.close()

            return result[0] if result else None
        except Exception as e:
            print(f"Error validating session: {e}")
            return None

    def logout_user(self, session_id):
        """Invalidate user session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM user_sessions WHERE session_id = ?', (session_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging out: {e}")
            return False

    def get_user_preferences(self, username):
        """Get user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT favorite_genres, preferred_platforms, rating_threshold 
                FROM user_preferences WHERE username = ?
            ''', (username,))

            result = cursor.fetchone()
            conn.close()

            if result:
                # Safely evaluate the string lists
                favorite_genres = []
                preferred_platforms = []

                try:
                    if result[0] and result[0] != '[]':
                        favorite_genres = eval(result[0])
                except:
                    favorite_genres = []

                try:
                    if result[1] and result[1] != '[]':
                        preferred_platforms = eval(result[1])
                except:
                    preferred_platforms = []

                return {
                    'favorite_genres': favorite_genres,
                    'preferred_platforms': preferred_platforms,
                    'rating_threshold': result[2] or 3.5
                }
            return {
                'favorite_genres': [],
                'preferred_platforms': [],
                'rating_threshold': 3.5
            }
        except Exception as e:
            print(f"Error getting preferences: {e}")
            return {
                'favorite_genres': [],
                'preferred_platforms': [],
                'rating_threshold': 3.5
            }

    def update_user_preferences(self, username, preferences):
        """Update user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences 
                (username, favorite_genres, preferred_platforms, rating_threshold, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                username,
                str(preferences.get('favorite_genres', [])),
                str(preferences.get('preferred_platforms', [])),
                preferences.get('rating_threshold', 3.5),
                datetime.now()
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating preferences: {e}")
            return False


def initialize_auth():
    """Initialize authentication system"""
    if 'auth_system' not in st.session_state:
        st.session_state.auth_system = AuthenticationSystem()
    return st.session_state.auth_system


def require_auth():
    """Decorator to require authentication for specific functions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            auth_system = initialize_auth()

            if 'user_session' not in st.session_state or not auth_system.validate_session(
                    st.session_state.user_session):
                st.error("🔐 Please log in to access this feature.")
                show_auth_interface()
                st.stop()

            return func(*args, **kwargs)

        return wrapper

    return decorator


def show_auth_interface():
    """Show login/register interface"""
    auth_system = initialize_auth()

    st.title("🎮 Game Assistant")
    st.markdown("---")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        with st.form("login_form"):
            st.subheader("Login to Your Account")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_btn = st.form_submit_button("Login")

            if login_btn:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    is_valid, message = auth_system.verify_user(username, password)
                    if is_valid:
                        session_id = auth_system.create_session(username)
                        if session_id:
                            st.session_state.user_session = session_id
                            st.session_state.username = username
                            # Initialize user preferences
                            st.session_state.user_preferences = auth_system.get_user_preferences(username)
                            st.success(f"Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error("Failed to create session. Please try again.")
                    else:
                        st.error(message)

    with tab2:
        with st.form("register_form"):
            st.subheader("Create New Account")
            new_username = st.text_input("Choose Username", placeholder="Minimum 3 characters")
            new_email = st.text_input("Email", placeholder="your.email@example.com")
            new_password = st.text_input("Choose Password", type="password", placeholder="Minimum 6 characters")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
            register_btn = st.form_submit_button("Create Account")

            if register_btn:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("All fields are required")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                else:
                    success, message = auth_system.create_user(new_username, new_email, new_password)
                    if success:
                        st.success(message)
                        # Auto-login after registration
                        session_id = auth_system.create_session(new_username)
                        if session_id:
                            st.session_state.user_session = session_id
                            st.session_state.username = new_username
                            # Initialize empty preferences for new user
                            st.session_state.user_preferences = {
                                'favorite_genres': [],
                                'preferred_platforms': [],
                                'rating_threshold': 3.5
                            }
                            st.rerun()
                    else:
                        st.error(message)

# Remove the show_user_profile function from auth_system.py
# We'll handle it in app.py where the genre mapping is available