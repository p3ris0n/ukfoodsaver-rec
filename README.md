UKFoodSaver Recommendations API (Prototype)

A FastAPI-based recommendation system for food preferences, providing personalized food recommendations using collaborative filtering and popularity-based algorithms.

🚀 Features

- Personalized Recommendations: Get food recommendations tailored to individual users
- Popular Food Discovery: Browse popular food items for new users
- Search Functionality: Search for specific food items by cuisine type
- Cold Start Handling: Smart fallback to popular items for new users
- Real-time Health Monitoring: API health and system statistics

📊 Data Source

The system uses the UKFS test dataset containing:
- User preferences for various cuisine types
- Rating data on a 0-2 scale
- Diverse food categories including Mexican, American, Italian, Chinese, Japanese, and more

🛠️ API Endpoints

Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API welcome and available endpoints |
| `/health` | GET | System health check |
| `/available-food` | GET | Get popular food items |
| `/for-you/{user_id}` | GET | Personalized recommendations for a user |
| `/search` | GET | Search food items by keyword |
| `/users` | GET | List all users in the system |
| `/items` | GET | List all food items in the system |

Debug Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/debug/data-source` | GET | Check data source and system status |
| `/debug/files` | GET | Check file structure and data file location |
| `/test-simple` | GET | Simple API connectivity test |

🎯 Usage Examples

Get Popular Food Items
```bash
curl "https://ukfoodsaver-rec.onrender.com/available-food?n=5"
```

Get Personalized Recommendations
```
curl "https://ukfoodsaver-rec.onrender.com/for-you/U1001?n=5"
```
Search Food Items
```
curl "https://ukfoodsaver-rec.onrender.com/search?query=mexican&n=10"
```

Check System Health
```
curl "https://ukfoodsaver-rec.onrender.com/health"
```

📋 Response Format

Available Food Endpoint
```json
{
  "type": "available_food",
  "recommendations": [
    {
      "item_id": "Mexican",
      "score": 0.94,
      "rank": 1
    }
  ],
  "count": 5
}
```

For You Endpoint
```json
{
  "type": "for_you",
  "user_id": "U1001",
  "recommendations": [
    {
      "item_id": "Italian",
      "score": 0.85,
      "rank": 1
    }
  ],
  "count": 5
}
```

🔧 Installation & Setup

Prerequisites
- Python 3.11+
- pip package manager

Local Development

1. Clone the repository
   ```
   git clone <repository-url>
   cd ukfoodsaver-rec
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Run the API server
   ```
   python render_api.py
   ```

4. Access the API
   - API: http://localhost:10000
   - Documentation: http://localhost:10000/docs

Deployment on Render

1. Connect your GitHub repository to Render
2. Set the following environment variables:
   - Start Command: `uvicorn render_api:app --host 0.0.0.0 --port $PORT`
   - Python Version: 3.11.10

3. Ensure your project structure includes:
   ```
   ├── render_api.py
   ├── requirements.txt
   ├── runtime.txt
   └── data/
       └── UKFS_testdata.csv
   ```

📁 Project Structure

```
ukfoodsaver-rec/
├── render_api.py          # Main FastAPI application
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version specification
├── data/
│   └── UKFS_testdata.csv # Food preference dataset
└── README.md            # This file
```

🎪 Algorithm Details

Recommendation Methods

1. *opularity-Based: For new users or cold start scenarios
   - Combines interaction frequency and average ratings
   - Normalized scoring between 0-1

2. Collaborative Filtering: For existing users
   - Finds users with similar preferences
   - Recommends items liked by similar users
   - Considers both frequency and rating scores

3. Hybrid Approach: Combines both methods for balanced recommendations

Scoring System
- Interaction Count: 70% weight
- Average Rating: 30% weight
- Final Score: Normalized to 0-1 range

🐛 Troubleshooting

Common Issues

1. 404 Errors
   - Check if the API is running: `curl /test-simple`
   - Verify endpoint names and parameters

2. 500 Internal Server Errors
   - Check Render logs for detailed error messages
   - Verify data file exists and is accessible

3. Data Loading Issues
   - Use `/debug/files` to check file structure
   - Use `/debug/data-source` to verify data loading

Debug Endpoints

```bash
# Check file structure
curl https://ukfoodsaver-rec.onrender.com/debug/files

# Check data source
curl https://ukfoodsaver-rec.onrender.com/debug/data-source

# Simple connectivity test
curl https://ukfoodsaver-rec.onrender.com/test-simple
```

📈 Performance Metrics

The system provides:
- Real-time health monitoring
- User and item statistics
- Recommendation quality metrics
- System performance tracking

🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

📞 Support

For support and questions:
- Check the API documentation at `/docs`
- Review the debug endpoints for system status
- Check Render deployment logs for errors

Live API: https://ukfoodsaver-rec.onrender.com

API Documentation: https://ukfoodsaver-rec.onrender.com/docs
