import csv
import sqlite3
from datetime import datetime
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def adapt_datetime(dt):
    return dt.isoformat()

def convert_datetime(s):
    return datetime.fromisoformat(s.decode())

# Register the adapter and converter
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("timestamp", convert_datetime)

def init_db():
    try:
        conn = sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES)
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
        conn.close()
        logging.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error initializing database: {str(e)}")

def import_data():
    # Initialize database tables
    init_db()

    try:
        conn = sqlite3.connect('ecommerce.db', detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()

        # Import customers
        with open('customers.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    # Clean up the data
                    browsing_history = row['Browsing_History'].strip('[]').replace("'", "").split(', ')
                    purchase_history = row['Purchase_History'].strip('[]').replace("'", "").split(', ')

                    # Generate unique email using customer_id
                    email = f"{row['customer_id']}@ecommerce.com"

                    c.execute('''INSERT INTO users (username, password, email, created_at, age, gender, location,
                                    browsing_history, purchase_history, customer_segment, avg_order_value, holiday, season)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (row['customer_id'], 'password123', email, datetime.now(),
                                int(row['Age']), row['Gender'], row['Location'],
                                str(browsing_history), str(purchase_history),
                                row['Customer_Segment'], float(row['Avg_Order_Value']),
                                row['Holiday'], row['Season']))
                    conn.commit()
                except sqlite3.IntegrityError as e:
                    logging.error(f"Error inserting user {row['customer_id']}: {str(e)}")
                    conn.rollback()  # Rollback the transaction to prevent partial inserts

        # Import products
        with open('products.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    # Clean up the data
                    similar_products = row['Similar_Product_List'].strip('[]').replace("'", "").split(', ')

                    c.execute('''INSERT INTO products (name, description, price, category, image_url,
                                    subcategory, brand, avg_rating, product_rating, sentiment_score,
                                    holiday, season, location, similar_products, recommendation_probability)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (f"{row['Category']} - {row['Subcategory']}",
                                f"Brand: {row['Brand']}, Location: {row['Geographical_Location']}",
                                float(row['Price']), row['Category'],
                                f"https://via.placeholder.com/300x200?text={row['product_id']}",
                                row['Subcategory'], row['Brand'],
                                float(row['Average_Rating_of_Similar_Products']),
                                float(row['Product_Rating']),
                                float(row['Customer_Review_Sentiment_Score']),
                                row['Holiday'], row['Season'], row['Geographical_Location'],
                                str(similar_products),
                                float(row['Probability_of_Recommendation'])))
                    conn.commit()
                except sqlite3.IntegrityError as e:
                    logging.error(f"Error inserting product {row['product_id']}: {str(e)}")
                    conn.rollback()  # Rollback the transaction to prevent partial inserts
                except ValueError as e:
                    logging.error(f"ValueError inserting product {row['product_id']}: {str(e)}")
                    conn.rollback()  # Rollback the transaction to prevent partial inserts

        logging.info("Data imported successfully.")
    except Exception as e:
        logging.error(f"Error importing data: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    import_data()
