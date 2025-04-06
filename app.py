from flask import Flask, render_template, request, jsonify
from recommender import parse_query_with_gemini, recommend_assessments
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """Render the main page with the query form"""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Process the query and return recommendations (form-based)"""
    query = request.form.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Use your existing recommender code
    filters = parse_query_with_gemini(query)
    results = recommend_assessments(filters)
    
    return jsonify({
        "filters": filters,
        "recommendations": results
    })

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    """REST API endpoint for recommendations (query parameter-based)"""
    query = request.args.get('query', '')
    
    if not query:
        return jsonify({
            "status": "error",
            "message": "Missing required 'query' parameter"
        }), 400
    
    try:
        # Use your existing recommender code
        filters = parse_query_with_gemini(query)
        results = recommend_assessments(filters)
        
        return jsonify({
            "status": "success",
            "filters": filters,
            "recommendations": results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)