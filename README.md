# DSmart - Multi-Agent E-commerce Recommendation System

DSmart is an intelligent e-commerce platform that uses a multi-agent system to provide personalized product recommendations to users. The system adapts to user preferences and behavior over time, offering increasingly relevant suggestions.

## Features

### 1. Multi-Agent Recommendation System

- **UserBehaviorAnalyzerAgent**: Analyzes user interactions and browsing patterns
- **ProductProfilerAgent**: Handles product embeddings and characteristics
- **RecommendationEngineAgent**: Generates personalized recommendations
- **FeedbackLoopAgent**: Updates recommendation weights based on user feedback
- **LongTermMemoryManagerAgent**: Manages user preferences over time

### 2. Smart Personalization

- Demographic-based recommendations (age, gender)
- Location-based suggestions
- Seasonal and holiday-specific recommendations
- Price range optimization based on user's average order value
- Category preferences learning

### 3. User Experience

- Modern, responsive UI with smooth animations
- Real-time product filtering and sorting
- Interactive product cards with hover effects
- Detailed product pages with similar item suggestions
- User feedback system for continuous improvement

### 4. Product Management

- Comprehensive product categorization
- Detailed product profiles including:
  - Ratings and reviews
  - Sentiment analysis
  - Price information
  - Brand details
  - Location availability
  - Seasonal availability

## Technical Stack

- **Backend**: Python Flask
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript
- **UI Framework**: Bootstrap 5
- **Icons**: Font Awesome
- **AI/ML**: NLTK for text processing
- **Recommendation Engine**: Custom multi-agent system

## Setup and Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd multiAgentPerplexityVigneshFrontend
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

3. Initialize the database:

```bash
python import_data.py
```

4. Run the application:

```bash
python app.py
```

5. Access the application at `http://127.0.0.1:5000`

## Project Structure

```
multiAgentPerplexityVigneshFrontend/
├── app.py                 # Main application file
├── import_data.py         # Database initialization and data import
├── requirements.txt       # Python dependencies
├── static/               # Static assets
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── images/          # Image assets
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── index.html       # Home page
│   ├── login.html       # Login page
│   ├── explore.html     # Product exploration page
│   └── product_detail.html # Product details page
└── ecommerce.db         # SQLite database
```

## Usage

1. **User Registration/Login**

   - New users can register with basic information
   - Existing users can log in to access personalized recommendations

2. **Product Exploration**

   - Browse products by category
   - Use filters for price, rating, and other attributes
   - Sort products by various criteria

3. **Product Interaction**

   - View detailed product information
   - See similar product recommendations
   - Provide feedback on recommendations
   - Add products to cart or wishlist

4. **Personalization**
   - System learns from user interactions
   - Recommendations improve over time
   - Feedback influences future suggestions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GeeksforGeeks Hackathon
- Contributors and maintainers
- Open source community
