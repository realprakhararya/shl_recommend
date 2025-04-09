from flask import Flask, render_template, request, jsonify
from recommender import parse_query_with_gemini, recommend_assessments
import os
import re

app = Flask(__name__)

# Load the title => url mapping from final.txt
def load_url_mappings():
    mappings = {}
    try:
        if os.path.exists("final.txt"):
            with open("final.txt", "r") as file:
                for line in file:
                    line = line.strip()
                    if "=>" in line:
                        title, url = line.split("=>", 1)
                        mappings[title.strip()] = url.strip()
    except Exception as e:
        print(f"Error loading URL mappings: {e}")
    return mappings

# Cache the mappings
URL_MAPPINGS = load_url_mappings()

# Test type mapping
TEST_TYPE_MAP = {
    "A": ["Ability & Aptitude"],
    "B": ["Biodata & Situational Judgement"],
    "C": ["Competencies"],
    "D": ["Development & 360"],
    "E": ["Assessment Exercises"],
    "K": ["Knowledge & Skills"],
    "P": ["Personality & Behavior"],
    "S": ["Simulations"]
}

@app.route('/', methods=['GET'])
def index():
    """Render the main page with the query form"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "healthy"})

@app.route('/recommend', methods=['POST'])
def recommend():
    """Process the query and return recommendations (form-based or JSON-based)"""
    # Check if request is JSON or form
    if request.is_json:
        data = request.get_json()
        query = data.get('query', '')
    else:
        query = request.form.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Use your existing recommender code
    filters = parse_query_with_gemini(query)
    results = recommend_assessments(filters)
    
    # Transform results to match the required response format
    recommended_assessments = []
    
    for result in results[:10]:  # Limit to at most 10 recommendations
        title = result["title"]
        
        # Get URL from mappings or create a fallback URL
        slug = re.sub(r'[^\w-]', '-', title.lower())
        url = URL_MAPPINGS.get(title, "https://www.shl.com/solutions/products/product-catalog/view/" + slug + "/")
        
        # Map test type code to descriptive value
        test_type_code = result.get("test_type", "M")
        test_type = TEST_TYPE_MAP.get(test_type_code, ["Mixed"])
        
        # Handle duration - default to 0 if not available or invalid
        try:
            duration = int(result.get("assessment_length", 0))
        except (ValueError, TypeError):
            duration = 0
        
        # Create assessment object
        assessment = {
            "url": url,
            "adaptive_support": "Yes" if result.get("adaptive/irt", "").lower() == "yes" else "No",
            "description": result.get("description", ""),
            "duration": duration,
            "remote_support": "Yes" if result.get("remote_testing", "").lower() == "yes" else "No",
            "test_type": test_type
        }
        
        recommended_assessments.append(assessment)
    
    # Ensure we have at least one recommendation
    if not recommended_assessments:
        # Create a generic recommendation if none are found
        recommended_assessments = [{
            "url": "https://www.shl.com/solutions/products/product-catalog/view/general-assessment/",
            "adaptive_support": "No",
            "description": "General assessment for evaluating candidate skills.",
            "duration": 30,
            "remote_support": "Yes",
            "test_type": ["Mixed"]
        }]
    
    return jsonify({"recommended_assessments": recommended_assessments})

if __name__ == "__main__":
    # Print startup message with loaded URL mappings
    print(f"Starting Flask server with {len(URL_MAPPINGS)} URL mappings loaded.")
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
