from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import sqlite3
import json
import ollama
from datetime import datetime
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import secrets
from werkzeug.utils import secure_filename
from flask_oauthlib.client import OAuth
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('punkt')
except Exception as e:
    logging.error(f"Error downloading NLTK data: {str(e)}")

# File upload configuration
UPLOAD_FOLDER = 'static/images/products'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def adapt_datetime(dt):
    return dt.isoformat()

def convert_datetime(s):
    return datetime.fromisoformat(s.decode())

# Register the adapter and converter
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("timestamp", convert_datetime)

app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static')
app.secret_key = secrets.token_hex(32)  # Generate a secure random key
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def get_db():
    """Get database connection with proper error handling"""
    try:
        conn = sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {str(e)}")
        raise

def login_required(f):
    """Decorator to check if user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def validate_input(data, required_fields):
    """Validate input data"""
    for field in required_fields:
        if field not in data or not data[field]:
            return False
    return True

# Database initialization
def init_db():
    try:
        with get_db() as conn:
            c = conn.cursor()

            # Create users table with additional fields
            c.execute('''CREATE TABLE IF NOT EXISTS users
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT UNIQUE,
            created_at timestamp,
            age INTEGER,
            gender TEXT,
            location TEXT,
            browsing_history TEXT,
            purchase_history TEXT,
            customer_segment TEXT,
            avg_order_value REAL,
            holiday TEXT,
            season TEXT)''')

            # Create products table with additional fields
            c.execute('''CREATE TABLE IF NOT EXISTS products
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            description TEXT,
            price REAL,
            category TEXT,
            image_url TEXT,
            subcategory TEXT,
            brand TEXT,
            avg_rating REAL,
            product_rating REAL,
            sentiment_score REAL,
            holiday TEXT,
            season TEXT,
            location TEXT,
            similar_products TEXT,
            recommendation_probability REAL)''')

            # Create user_behavior table
            c.execute('''CREATE TABLE IF NOT EXISTS user_behavior
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            product_id INTEGER,
            action_type TEXT,
            timestamp timestamp,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (product_id) REFERENCES products (id))''')

            # Create feedback table to store user feedback
            c.execute('''CREATE TABLE IF NOT EXISTS product_feedback
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            product_id INTEGER,
            feedback_type TEXT,
            timestamp timestamp,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (product_id) REFERENCES products (id))''')

            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {str(e)}")
        raise

# Initialize database
init_db()

# Agent Definitions
class UserBehaviorAnalyzerAgent:
    def process_user_behavior(self, user_id):
        try:
            conn = sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES)
            c = conn.cursor()
            c.execute('''SELECT p.name, ub.action_type
                         FROM user_behavior ub
                         JOIN products p ON ub.product_id = p.id
                         WHERE ub.user_id = ?
                         ORDER BY ub.timestamp DESC LIMIT 10''', (user_id,))
            behavior_data = c.fetchall()
            conn.close()
            # Process behavior data to extract user intent
            intent_signals = self.extract_intent_signals(behavior_data)
            return intent_signals
        except sqlite3.Error as e:
            logging.error(f"Error processing user behavior for user {user_id}: {str(e)}")
            return []

    def extract_intent_signals(self, behavior_data):
        try:
            # Implement logic to extract intent (e.g., keywords, categories)
            intent_signals = []
            for product_name, action_type in behavior_data:
                if action_type == 'view':
                    intent_signals.append(f"Viewed {product_name}")
                elif action_type == 'purchase':
                    intent_signals.append(f"Purchased {product_name}")

            # Tokenize and remove stop words
            stop_words = set(stopwords.words('english'))
            tokens = []
            for signal in intent_signals:
                word_tokens = word_tokenize(signal)
                filtered_tokens = [w for w in word_tokens if not w in stop_words]
                tokens.extend(filtered_tokens)

            # Get the most frequent keywords
            keyword_counts = {}
            for token in tokens:
                keyword_counts[token] = keyword_counts.get(token, 0) + 1
            most_frequent_keywords = sorted(keyword_counts, key=keyword_counts.get, reverse=True)[:5]

            return most_frequent_keywords
        except Exception as e:
            logging.error(f"Error extracting intent signals: {str(e)}")
            return []

class ProductProfilerAgent:
    def get_product_embeddings(self, product_ids):
        try:
            conn = sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES)
            c = conn.cursor()
            embeddings = {}
            for product_id in product_ids:
                c.execute('''SELECT category, subcategory, brand, description FROM products WHERE id = ?''', (product_id,))
                product_data = c.fetchone()
                if product_data:
                    category, subcategory, brand, description = product_data
                    # Combine features into a single embedding string
                    embedding_string = f"{category} {subcategory} {brand} {description}"
                    embeddings[product_id] = embedding_string
            conn.close()
            return embeddings
        except sqlite3.Error as e:
            logging.error(f"Error getting product embeddings for product IDs {product_ids}: {str(e)}")
            return {}

class RecommendationEngineAgent:
    def __init__(self):
        self.model = "mistral"

    def generate_recommendations(self, user_id, user_intent, product_embeddings):
        try:
            # Get user's profile and browsing history
            conn = sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES)
            c = conn.cursor()

            # Get user profile
            c.execute('''SELECT age, gender, location, browsing_history, purchase_history,
                        customer_segment, avg_order_value, holiday, season
                        FROM users WHERE id = ?''', (user_id,))

            user_profile = c.fetchone()
            conn.close()

            if not user_profile:
                return "No user profile found."

            context = {
                'user_profile': {
                    'age': user_profile[0],
                    'gender': user_profile[1],
                    'location': user_profile[2],
                    'browsing_history': eval(user_profile[3]) if user_profile[3] else [],
                    'purchase_history': eval(user_profile[4]) if user_profile[4] else [],
                    'segment': user_profile[5],
                    'avg_order_value': user_profile[6],
                    'holiday': user_profile[7],
                    'season': user_profile[8]
                },
                'user_intent': user_intent,
                'product_embeddings': product_embeddings
            }
            prompt = f"""Based on this user profile: {context['user_profile']} and user intent keywords: {context['user_intent']}, and the product details {context['product_embeddings']}, suggest 5 relevant products. Provide product names and brief explanations."""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            recommendations = response["message"]["content"]
            return recommendations
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            return "Unable to generate recommendations at this time."

class FeedbackLoopAgent:
    def update_recommendation_weights(self, user_id, product_id, feedback):
        try:
            conn = sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES)
            c = conn.cursor()

            # Record the feedback
            c.execute('''INSERT INTO product_feedback (user_id, product_id, feedback_type, timestamp) 
                         VALUES (?, ?, ?, ?)''', (user_id, product_id, feedback, datetime.now()))

            # Adjust recommendation probability based on feedback
            if feedback == 'positive':
                c.execute('''UPDATE products SET recommendation_probability = 
                             CASE 
                                 WHEN recommendation_probability IS NULL THEN 0.1
                                 ELSE MIN(recommendation_probability + 0.1, 1.0) 
                             END
                             WHERE id = ?''', (product_id,))
            elif feedback == 'negative':
                c.execute('''UPDATE products SET recommendation_probability = 
                             CASE 
                                 WHEN recommendation_probability IS NULL THEN -0.1
                             ELSE MAX(recommendation_probability - 0.1, 0.0)
                             END
                             WHERE id = ?''', (product_id,))

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Error updating recommendation weights for user {user_id}, product {product_id}: {str(e)}")

class LongTermMemoryManagerAgent:
    def get_user_preferences(self, user_id):
        try:
            conn = sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES)
            c = conn.cursor()
            c.execute('''SELECT age, gender, location, browsing_history, purchase_history, 
                         customer_segment, avg_order_value, holiday, season
                         FROM users WHERE id = ?''', (user_id,))
            user_data = c.fetchone()
            conn.close()
            if user_data:
                (age, gender, location, browsing_history, purchase_history,
                 customer_segment, avg_order_value, holiday, season) = user_data
                preferences = {
                    'age': age,
                    'gender': gender,
                    'location': location,
                    'browsing_history': eval(browsing_history) if browsing_history else [],
                    'purchase_history': eval(purchase_history) if purchase_history else [],
                    'customer_segment': customer_segment,
                    'avg_order_value': avg_order_value,
                    'holiday': holiday,
                    'season': season
                }
                return preferences
            return None
        except sqlite3.Error as e:
            logging.error(f"Error getting user preferences for user {user_id}: {str(e)}")
            return None

def get_personalized_products(user_id):
    conn = sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()

    # Get user preferences
    c.execute('''SELECT age, gender, location, browsing_history, purchase_history,
                 customer_segment, avg_order_value, holiday, season
                 FROM users WHERE id = ?''', (user_id,))
    user_data = c.fetchone()

    if not user_data:
        return []

    # Extract user preferences
    age = user_data[0]
    gender = user_data[1]
    location = user_data[2]
    browsing_history = eval(user_data[3]) if user_data[3] else []
    purchase_history = eval(user_data[4]) if user_data[4] else []
    customer_segment = user_data[5]
    avg_order_value = user_data[6]
    holiday = user_data[7]
    season = user_data[8]

    # Build personalized query based on user preferences
    query = '''SELECT id, name, description, price, category, image_url,
                    subcategory, brand, avg_rating, product_rating, sentiment_score
             FROM products WHERE 1=1'''
    params = []

    # Filter by location if specified
    if location:
        query += " AND location = ?"
        params.append(location)

    # Filter by season if specified
    if season:
        query += " AND season = ?"
        params.append(season)

    # Filter by holiday if specified
    if holiday:
        query += " AND holiday = ?"
        params.append(holiday)

    # Filter by price range based on average order value
    if avg_order_value:
        min_price = avg_order_value * 0.5
        max_price = avg_order_value * 1.5
        query += " AND price BETWEEN ? AND ?"
        params.extend([min_price, max_price])

    # Add browsing history products
    if browsing_history:
        placeholders = ','.join(['?'] * len(browsing_history))
        query += f" OR id IN ({placeholders})"
        params.extend(browsing_history)

    # Add purchase history products
    if purchase_history:
        placeholders = ','.join(['?'] * len(purchase_history))
        query += f" OR id IN ({placeholders})"
        params.extend(purchase_history)

    # Add gender-specific products if gender is specified
    if gender:
        if gender.lower() == 'male':
            query += " OR category IN ('Electronics', 'Sports', 'Books')"
        elif gender.lower() == 'female':
            query += " OR category IN ('Fashion', 'Beauty', 'Home')"

    # Add age-specific products
    if age:
        if age < 25:
            query += " OR category IN ('Electronics', 'Fashion', 'Sports')"
        elif 25 <= age < 40:
            query += " OR category IN ('Home', 'Beauty', 'Electronics')"
        else:
            query += " OR category IN ('Home', 'Books', 'Beauty')"

    # Execute the query
    c.execute(query, params)
    products = c.fetchall()

    conn.close()
    return products

# Initialize Agents
user_behavior_analyzer = UserBehaviorAnalyzerAgent()
product_profiler = ProductProfilerAgent()
recommendation_engine = RecommendationEngineAgent()
feedback_loop = FeedbackLoopAgent()
long_term_memory_manager = LongTermMemoryManagerAgent()

# OAuth setup
oauth = OAuth(app)

# Google OAuth configuration
google = oauth.remote_app(
    'google',
    consumer_key=os.environ.get('GOOGLE_CLIENT_ID', 'your-google-client-id'),
    consumer_secret=os.environ.get('GOOGLE_CLIENT_SECRET', 'your-google-client-secret'),
    request_token_params={
        'scope': 'email profile'
    },
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)

# Apple OAuth configuration
apple = oauth.remote_app(
    'apple',
    consumer_key=os.environ.get('APPLE_CLIENT_ID', 'your-apple-client-id'),
    consumer_secret=os.environ.get('APPLE_CLIENT_SECRET', 'your-apple-client-secret'),
    request_token_params={
        'scope': 'email name',
        'response_type': 'code id_token'
    },
    base_url='https://appleid.apple.com/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://appleid.apple.com/auth/token',
    authorize_url='https://appleid.apple.com/auth/authorize',
)

# Routes
@app.route('/')
def index():
    try:
        if 'user_id' not in session:
            return redirect(url_for('login'))

        user_id = session['user_id']

        # 1. Analyze User Behavior
        user_intent = user_behavior_analyzer.process_user_behavior(user_id)

        # 2. Get Product Embeddings
        recent_product_ids = [item[0] for item in sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES).cursor().execute(
            '''SELECT product_id FROM user_behavior WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5''', (user_id,)).fetchall()]
        product_embeddings = product_profiler.get_product_embeddings(recent_product_ids)

        # 3. Generate Recommendations
        recommendations = recommendation_engine.generate_recommendations(user_id, user_intent, product_embeddings)
        products = get_personalized_products(user_id)

        return render_template('index.html', recommendations=recommendations, products=products)
    except Exception as e:
        logging.error(f"Error in index route: {str(e)}")
        flash("An error occurred. Please try again.", "error")
        return render_template('index.html', recommendations=None, products=[])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            if not validate_input(request.form, ['username', 'password']):
                flash('Please provide both username and password.', 'error')
                return render_template('login.html')

            username = request.form['username']
            password = request.form['password']

            with get_db() as conn:
                c = conn.cursor()
                c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
                user = c.fetchone()

                if user and check_password_hash(user['password'], password):
                    session['user_id'] = user['id']
                    flash('Successfully logged in!', 'success')
                    return redirect(url_for('index'))
                else:
                    flash('Invalid username or password.', 'error')
            return render_template('login.html')
        except sqlite3.Error as e:
            logging.error(f"Database error during login: {str(e)}")
            flash('An error occurred. Please try again.', 'error')
            return render_template('login.html')
        except Exception as e:
            logging.error(f"Unexpected error during login: {str(e)}")
            flash('An unexpected error occurred. Please try again.', 'error')
            return render_template('login.html')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            required_fields = ['username', 'password', 'email']
            if not validate_input(request.form, required_fields):
                flash('Please fill in all required fields.', 'error')
                return render_template('register.html')

            username = request.form['username']
            password = request.form['password']
            email = request.form['email']

            # Hash the password
            hashed_password = generate_password_hash(password)

            with get_db() as conn:
                c = conn.cursor()
                try:
                    c.execute('''INSERT INTO users (username, password, email, created_at) 
                                VALUES (?, ?, ?, ?)''',
                             (username, hashed_password, email, datetime.now()))
                    conn.commit()
                    flash('Registration successful! Please login.', 'success')
                    return redirect(url_for('login'))
                except sqlite3.IntegrityError:
                    flash('Username or email already exists.', 'error')
            return render_template('register.html')
        except sqlite3.Error as e:
            logging.error(f"Database error during registration: {str(e)}")
            flash('An error occurred. Please try again.', 'error')
            return render_template('register.html')
        except Exception as e:
            logging.error(f"Unexpected error during registration: {str(e)}")
            flash('An unexpected error occurred. Please try again.', 'error')
            return render_template('register.html')

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/product/<int:product_id>')
@login_required
def product_detail(product_id):
    try:
        with get_db() as conn:
            c = conn.cursor()

        # Record user view
            c.execute('''INSERT INTO user_behavior (user_id, product_id, action_type, timestamp) 
                        VALUES (?, ?, ?, ?)''',
                     (session['user_id'], product_id, 'view', datetime.now()))
            
            # Get product details
            c.execute('''SELECT * FROM products WHERE id = ?''', (product_id,))
            product = c.fetchone()
            
            if not product:
                flash('Product not found.', 'error')
                return redirect(url_for('index'))
            
            # Get similar products
            c.execute('''SELECT * FROM products 
                        WHERE category = ? AND id != ? 
                        LIMIT 5''', (product['category'], product_id))
            similar_products = c.fetchall()
            
            conn.commit()
            
            # Convert to dictionary for template
            product_dict = dict(product)
            similar_products_dict = [dict(p) for p in similar_products]
            
            # Log for debugging
            logging.info(f"Product data: {product_dict}")
            logging.info(f"Similar products: {similar_products_dict}")
            
            return render_template('product_detail.html', 
                                 product=product_dict, 
                                 similar_products=similar_products_dict)
            
    except sqlite3.Error as e:
        logging.error(f"Database error in product detail: {str(e)}")
        flash('An error occurred while loading the product.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Unexpected error in product detail: {str(e)}")
        flash('An unexpected error occurred.', 'error')
        return redirect(url_for('index'))

@app.route('/api/product/<int:product_id>/feedback', methods=['POST'])
@login_required
def product_feedback(product_id):
    try:
        if not validate_input(request.json, ['feedback_type']):
            return jsonify({'error': 'Invalid feedback data'}), 400

        feedback_type = request.json['feedback_type']
        if feedback_type not in ['positive', 'negative']:
            return jsonify({'error': 'Invalid feedback type'}), 400

        with get_db() as conn:
            c = conn.cursor()
            
            # Record feedback
            c.execute('''INSERT INTO product_feedback 
                        (user_id, product_id, feedback_type, timestamp) 
                        VALUES (?, ?, ?, ?)''',
                     (session['user_id'], product_id, feedback_type, datetime.now()))
            
            # Update product rating
            if feedback_type == 'positive':
                c.execute('''UPDATE products 
                            SET product_rating = COALESCE(product_rating, 0) + 0.1 
                            WHERE id = ?''', (product_id,))
            else:
                c.execute('''UPDATE products 
                            SET product_rating = GREATEST(COALESCE(product_rating, 0) - 0.1, 0) 
                            WHERE id = ?''', (product_id,))
            
            conn.commit()
            return jsonify({'message': 'Feedback recorded successfully'})
    except sqlite3.Error as e:
        logging.error(f"Database error in product feedback: {str(e)}")
        return jsonify({'error': 'Database error occurred'}), 500
    except Exception as e:
        logging.error(f"Unexpected error in product feedback: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/admin/products')
@login_required
def admin_products():
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM products ORDER BY id DESC')
            products = c.fetchall()
            return render_template('admin/products.html', products=[dict(p) for p in products])
    except Exception as e:
        logging.error(f"Error in admin products: {str(e)}")
        flash('An error occurred while loading products.', 'error')
        return redirect(url_for('index'))

@app.route('/admin/products/add', methods=['GET', 'POST'])
@login_required
def admin_add_product():
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form.get('name')
            description = request.form.get('description')
            price = float(request.form.get('price', 0))
            category = request.form.get('category')
            subcategory = request.form.get('subcategory')
            brand = request.form.get('brand')
            
            # Handle image upload
            image_url = None
            if 'image' in request.files:
                file = request.files['image']
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    # Add timestamp to filename to make it unique
                    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    # Use absolute path for image URL
                    image_url = f"/static/images/products/{filename}"
                    logging.info(f"Saved image to: {file_path}, URL: {image_url}")
            
            with get_db() as conn:
                c = conn.cursor()
                c.execute('''INSERT INTO products 
                            (name, description, price, category, image_url, subcategory, brand, avg_rating, product_rating) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)''',
                         (name, description, price, category, image_url, subcategory, brand))
                conn.commit()
                flash('Product added successfully!', 'success')
                return redirect(url_for('admin_products'))
                
        except Exception as e:
            logging.error(f"Error adding product: {str(e)}")
            flash('An error occurred while adding the product.', 'error')
            return redirect(url_for('admin_add_product'))
    
    return render_template('admin/add_product.html')

@app.route('/admin/products/edit/<int:product_id>', methods=['GET', 'POST'])
@login_required
def admin_edit_product(product_id):
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            if request.method == 'POST':
                # Get form data
                name = request.form.get('name')
                description = request.form.get('description')
                price = float(request.form.get('price', 0))
                category = request.form.get('category')
                subcategory = request.form.get('subcategory')
                brand = request.form.get('brand')
                
                # Handle image upload
                image_url = request.form.get('current_image')
                if 'image' in request.files:
                    file = request.files['image']
                    if file and file.filename and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        # Add timestamp to filename to make it unique
                        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        image_url = f"/static/images/products/{filename}"
                        logging.info(f"Updated image to: {file_path}, URL: {image_url}")
                
                c.execute('''UPDATE products 
                            SET name = ?, description = ?, price = ?, category = ?, 
                                image_url = ?, subcategory = ?, brand = ?
                            WHERE id = ?''',
                         (name, description, price, category, image_url, subcategory, brand, product_id))
                conn.commit()
                flash('Product updated successfully!', 'success')
                return redirect(url_for('admin_products'))
            
            # GET request - load product data
            c.execute('SELECT * FROM products WHERE id = ?', (product_id,))
            product = c.fetchone()
            
            if not product:
                flash('Product not found.', 'error')
                return redirect(url_for('admin_products'))
                
            return render_template('admin/edit_product.html', product=dict(product))
            
    except Exception as e:
        logging.error(f"Error editing product: {str(e)}")
        flash('An error occurred while editing the product.', 'error')
        return redirect(url_for('admin_products'))

@app.route('/admin/products/delete/<int:product_id>', methods=['POST'])
@login_required
def admin_delete_product(product_id):
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # Get product image path before deleting
            c.execute('SELECT image_url FROM products WHERE id = ?', (product_id,))
            product = c.fetchone()
            
            # Delete product from database
            c.execute('DELETE FROM products WHERE id = ?', (product_id,))
            conn.commit()
            
            # Delete image file if it exists
            if product and product['image_url']:
                image_path = os.path.join(app.root_path, product['image_url'].lstrip('/'))
                if os.path.exists(image_path):
                    os.remove(image_path)
            
            flash('Product deleted successfully!', 'success')
            
    except Exception as e:
        logging.error(f"Error deleting product: {str(e)}")
        flash('An error occurred while deleting the product.', 'error')
    
    return redirect(url_for('admin_products'))

@app.route('/explore')
@login_required
def explore():
    try:
        # Get query parameters
        search_query = request.args.get('search', '')
        current_category = request.args.get('category', 'all')
        current_sort = request.args.get('sort', 'rating')
        min_price = request.args.get('min_price', '')
        max_price = request.args.get('max_price', '')
        
        with get_db() as conn:
            c = conn.cursor()
            
            # Base query
            query = '''SELECT * FROM products WHERE 1=1'''
            params = []
            
            # Apply search filter
            if search_query:
                query += ''' AND (name LIKE ? OR description LIKE ? OR category LIKE ?)'''
                search_param = f'%{search_query}%'
                params.extend([search_param, search_param, search_param])
            
            # Apply category filter
            if current_category != 'all':
                query += ''' AND category = ?'''
                params.append(current_category)
            
            # Apply price filter
            if min_price:
                query += ''' AND price >= ?'''
                params.append(float(min_price))
            if max_price:
                query += ''' AND price <= ?'''
                params.append(float(max_price))
            
            # Apply sorting
            if current_sort == 'price_low':
                query += ''' ORDER BY price ASC'''
            elif current_sort == 'price_high':
                query += ''' ORDER BY price DESC'''
            elif current_sort == 'newest':
                query += ''' ORDER BY id DESC'''
            else:  # default to rating
                query += ''' ORDER BY product_rating DESC, avg_rating DESC'''
            
            # Add limit
            query += ''' LIMIT 20'''
            
            # Execute query
            c.execute(query, params)
            products = c.fetchall()
            
            # Get categories for filtering
            c.execute('''SELECT DISTINCT category FROM products''')
            categories = [row['category'] for row in c.fetchall()]
            
            # Get user preferences for personalized recommendations
            c.execute('''SELECT age, gender, location, customer_segment 
                        FROM users WHERE id = ?''', (session['user_id'],))
            user_prefs = c.fetchone()
            
            return render_template('explore.html', 
                                 products=[dict(p) for p in products],
                                 categories=categories,
                                 user_prefs=dict(user_prefs) if user_prefs else None,
                                 search_query=search_query,
                                 current_category=current_category,
                                 current_sort=current_sort,
                                 min_price=min_price,
                                 max_price=max_price)
            
    except sqlite3.Error as e:
        logging.error(f"Database error in explore page: {str(e)}")
        flash('An error occurred while loading the explore page.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Unexpected error in explore page: {str(e)}")
        flash('An unexpected error occurred.', 'error')
        return redirect(url_for('index'))

@app.route('/test-static')
def test_static():
    return """
    <html>
        <head>
            <link rel="stylesheet" type="text/css" href="/static/css/style.css">
        </head>
        <body>
            <div class="welcome-banner">
                <h1>Test Banner</h1>
            </div>
            <img src="/static/images/products/20250406145046_book.jpg" alt="Test Image" style="max-width: 300px;">
            <p>Test file: <a href="/static/images/test.txt">test.txt</a></p>
        </body>
    </html>
    """

@app.route('/check-image/<filename>')
def check_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    exists = os.path.exists(file_path)
    return jsonify({
        'filename': filename,
        'file_path': file_path,
        'exists': exists,
        'absolute_path': os.path.abspath(file_path)
    })

@app.route('/login/google')
def google_login():
    return google.authorize(callback=url_for('google_authorized', _external=True))

@app.route('/login/google/authorized')
def google_authorized():
    resp = google.authorized_response()
    if resp is None or resp.get('access_token') is None:
        flash('Access denied: reason={} error={}'.format(
            request.args['error_reason'],
            request.args['error_description']
        ))
        return redirect(url_for('login'))
    
    session['google_token'] = (resp['access_token'], '')
    me = google.get('userinfo')
    
    # Extract user information
    email = me.data['email']
    name = me.data.get('name', '')
    picture = me.data.get('picture', '')
    
    # Check if user exists
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    
    if user is None:
        # Create new user
        username = email.split('@')[0]
        password = secrets.token_urlsafe(16)
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO users (username, email, password, name, profile_picture, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, email, hashed_password, name, picture, datetime.now()))
        conn.commit()
        
        # Get the new user's ID
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        user_id = cursor.fetchone()[0]
        
        # Create user profile
        cursor.execute('''
            INSERT INTO user_profiles (user_id, preferences, created_at)
            VALUES (?, ?, ?)
        ''', (user_id, json.dumps({}), datetime.now()))
        conn.commit()
        
        flash('Account created successfully! Welcome to NDUmart!')
    else:
        user_id = user[0]
    
    conn.close()
    
    # Set session
    session['user_id'] = user_id
    session['username'] = email
    return redirect(url_for('index'))

@app.route('/login/apple')
def apple_login():
    return apple.authorize(callback=url_for('apple_authorized', _external=True))

@app.route('/login/apple/authorized')
def apple_authorized():
    resp = apple.authorized_response()
    if resp is None or resp.get('access_token') is None:
        flash('Access denied: reason={} error={}'.format(
            request.args['error_reason'],
            request.args['error_description']
        ))
        return redirect(url_for('login'))
    
    session['apple_token'] = (resp['access_token'], '')
    
    # Get user info from Apple
    user_info = resp.get('id_token', {}).get('payload', {})
    email = user_info.get('email')
    name = user_info.get('name', {}).get('firstName', '') + ' ' + user_info.get('name', {}).get('lastName', '')
    
    if not email:
        flash('Could not retrieve email from Apple')
        return redirect(url_for('login'))
    
    # Check if user exists
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    
    if user is None:
        # Create new user
        username = email.split('@')[0]
        password = secrets.token_urlsafe(16)
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO users (username, email, password, name, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, hashed_password, name, datetime.now()))
        conn.commit()
        
        # Get the new user's ID
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        user_id = cursor.fetchone()[0]
        
        # Create user profile
        cursor.execute('''
            INSERT INTO user_profiles (user_id, preferences, created_at)
            VALUES (?, ?, ?)
        ''', (user_id, json.dumps({}), datetime.now()))
        conn.commit()
        
        flash('Account created successfully! Welcome to NDUmart!')
    else:
        user_id = user[0]
    
    conn.close()
    
    # Set session
    session['user_id'] = user_id
    session['username'] = email
    return redirect(url_for('index'))

@google.tokengetter
def get_google_oauth_token():
    return session.get('google_token')

@apple.tokengetter
def get_apple_oauth_token():
    return session.get('apple_token')

if __name__ == '__main__':
    app.run(debug=True)
