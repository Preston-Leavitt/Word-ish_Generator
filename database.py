import sqlite3
from datetime import datetime
from contextlib import contextmanager

DATABASE = 'wordish.db'

@contextmanager
def get_db():
    """Get database connection with context manager"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Initialize database schema"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Words table - stores all generated words
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                definition TEXT NOT NULL,
                is_new BOOLEAN NOT NULL,
                image TEXT,
                song TEXT,
                source TEXT,
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users (id)
            )
        ''')
        
        # Votes table - upvotes and downvotes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                vote_type INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(word_id, user_id),
                FOREIGN KEY (word_id) REFERENCES words (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_words_created_at ON words(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_words_word ON words(word)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_votes_word_id ON votes(word_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_votes_user_id ON votes(user_id)')

def get_user_by_username(username):
    """Get user by username"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        return cursor.fetchone()

def get_user_by_id(user_id):
    """Get user by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        return cursor.fetchone()

def create_user(username, password_hash):
    """Create new user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, password_hash)
        )
        return cursor.lastrowid

def save_word(word, definition, is_new, image, song, source, user_id):
    """Save a word to database"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO words (word, definition, is_new, image, song, source, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (word, definition, is_new, image, song, source, user_id))
        return cursor.lastrowid

def get_word_by_text(word_text):
    """Get word by text"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM words WHERE LOWER(word) = LOWER(?) ORDER BY created_at DESC LIMIT 1', (word_text,))
        return cursor.fetchone()

def get_vote(word_id, user_id):
    """Get user's vote for a word"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT vote_type FROM votes WHERE word_id = ? AND user_id = ?',
            (word_id, user_id)
        )
        result = cursor.fetchone()
        return result['vote_type'] if result else None

def set_vote(word_id, user_id, vote_type):
    """Set or update a vote (1 for upvote, -1 for downvote, 0 to remove)"""
    with get_db() as conn:
        cursor = conn.cursor()
        if vote_type == 0:
            cursor.execute(
                'DELETE FROM votes WHERE word_id = ? AND user_id = ?',
                (word_id, user_id)
            )
        else:
            cursor.execute('''
                INSERT INTO votes (word_id, user_id, vote_type)
                VALUES (?, ?, ?)
                ON CONFLICT(word_id, user_id) 
                DO UPDATE SET vote_type = excluded.vote_type, created_at = CURRENT_TIMESTAMP
            ''', (word_id, user_id, vote_type))

def get_word_score(word_id):
    """Get total score (upvotes - downvotes) for a word"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT COALESCE(SUM(vote_type), 0) as score FROM votes WHERE word_id = ?',
            (word_id,)
        )
        return cursor.fetchone()['score']

def get_leaderboard(time_filter='lifetime', limit=50):
    """Get leaderboard of top words
    time_filter: 'day', 'month', or 'lifetime'
    """
    time_clause = ""
    if time_filter == 'day':
        time_clause = "AND w.created_at >= datetime('now', '-1 day')"
    elif time_filter == 'month':
        time_clause = "AND w.created_at >= datetime('now', '-1 month')"
    
    query = f'''
        SELECT 
            w.id,
            w.word,
            w.definition,
            w.image,
            w.song,
            w.created_at,
            w.is_new,
            COALESCE(SUM(v.vote_type), 0) as score,
            COUNT(CASE WHEN v.vote_type = 1 THEN 1 END) as upvotes,
            COUNT(CASE WHEN v.vote_type = -1 THEN 1 END) as downvotes,
            u.username as creator
        FROM words w
        LEFT JOIN votes v ON w.id = v.word_id
        LEFT JOIN users u ON w.created_by = u.id
        WHERE 1=1 {time_clause}
        GROUP BY w.id
        ORDER BY score DESC, w.created_at DESC
        LIMIT ?
    '''
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (limit,))
        return cursor.fetchall()

def get_daily_words(limit=100):
    """Get all words created today"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                w.id,
                w.word,
                w.definition,
                w.created_at,
                w.is_new,
                COALESCE(SUM(v.vote_type), 0) as score,
                u.username as creator
            FROM words w
            LEFT JOIN votes v ON w.id = v.word_id
            LEFT JOIN users u ON w.created_by = u.id
            WHERE w.created_at >= date('now')
            GROUP BY w.id
            ORDER BY w.created_at DESC
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

def get_random_word():
    """Get a random word from the database"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                w.id,
                w.word,
                w.definition,
                w.image,
                w.song,
                w.created_at,
                w.is_new,
                COALESCE(SUM(v.vote_type), 0) as score,
                u.username as creator
            FROM words w
            LEFT JOIN votes v ON w.id = v.word_id
            LEFT JOIN users u ON w.created_by = u.id
            GROUP BY w.id
            ORDER BY RANDOM()
            LIMIT 1
        ''')
        return cursor.fetchone()

def search_words(query, limit=20):
    """Search words by text"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                w.id,
                w.word,
                w.definition,
                w.image,
                w.song,
                w.created_at,
                w.is_new,
                COALESCE(SUM(v.vote_type), 0) as score,
                u.username as creator
            FROM words w
            LEFT JOIN votes v ON w.id = v.word_id
            LEFT JOIN users u ON w.created_by = u.id
            WHERE LOWER(w.word) LIKE LOWER(?)
            GROUP BY w.id
            ORDER BY score DESC, w.created_at DESC
            LIMIT ?
        ''', (f'%{query}%', limit))
        return cursor.fetchall()

def get_user_words(user_id):
    """Get all words created by a specific user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                w.id,
                w.word,
                w.definition,
                w.image,
                w.song,
                w.created_at,
                w.is_new,
                COALESCE(SUM(v.vote_type), 0) as score
            FROM words w
            LEFT JOIN votes v ON w.id = v.word_id
            WHERE w.created_by = ?
            GROUP BY w.id
            ORDER BY w.created_at DESC
        ''', (user_id,))
        return cursor.fetchall()

def update_username(user_id, new_username):
    """Update user's username"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET username = ? WHERE id = ?', (new_username, user_id))
        return cursor.rowcount > 0

def update_password(user_id, new_password_hash):
    """Update user's password"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_password_hash, user_id))
        return cursor.rowcount > 0
