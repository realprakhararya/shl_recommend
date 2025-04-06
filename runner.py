import typer
from recommender import recommend_assessments, parse_query_with_gemini
import pandas as pd
import sys

app = typer.Typer()

# Optional: preload dataset for use in recommend_assessments if needed
# df = pd.read_csv("shl_clean.csv", encoding='latin1')

@app.command()
def prompt_debug(query: str):
    """Debug the Gemini-based filter extraction"""
    print("📨 User Query:", query)
    filters = parse_query_with_gemini(query)
    print("🧠 Extracted Filters:", filters)

@app.command()
def recommend(query: str):
    """Recommend assessments based on a natural language query"""
    filters = parse_query_with_gemini(query)
    results = recommend_assessments(filters)
    for r in results:
        print(f"✅ {r['title']} — Score: {r['score']} — Duration: {r['assessment_length']}")

@app.command()
def run_eval(output_file: str = "eval_results.json"):
    """Run a comprehensive evaluation of the recommender system and save results to a file"""
    import json
    from datetime import datetime
    
    # Optional: initialize LangSmith client only if needed for logging
    try:
        from langsmith import Client
        client = Client()
        use_langsmith = True
        dataset_name = f"shl-recommender-eval-{datetime.now().strftime('%Y%m%d')}"
        print(f"🚀 Running evaluation and saving to LangSmith dataset: {dataset_name}")
    except:
        use_langsmith = False
        print("🚀 Running evaluation without LangSmith integration")
    
    test_cases = [
        {
            "query": "Looking for assessments on Python and ML, 45 mins max",
            "expected_filters": {
                "skills": ["python", "machine learning"],
                "duration_limit": 45,
                "job_level": None
            }
        },
        {
            "query": "Need something for a research engineer on generative AI and NLP",
            "expected_filters": {
                "skills": ["ai", "nlp"],
                "job_level": "research engineer",
                "duration_limit": None
            }
        },
        {
            "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
            "expected_filters": {
                "skills": ["java", "collaboration"],
                "job_level": "developer",
                "duration_limit": 40
            }
        },
        {
            "query": "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
            "expected_filters": {
                "skills": ["python", "sql", "javascript"],
                "job_level": "mid",
                "duration_limit": 60
            }
        },
        {
            "query": "Are you an AI enthusiast with visionary thinking to conceptualize AI-based products? Are you looking to apply these skills in an environment where teamwork and collaboration are key to developing our digital product experiences? We are seeking a Research Engineer to join our team to deliver robust AI/ML models. You will closely work with the product team to spot opportunities to use AI in the current product stack and influence the product roadmap by incorporating AI-led features/products. Can you recommend some assessment that can help me screen applications? Time limit is less than 30 minutes.",
            "expected_filters": {
                "skills": ["ai", "ml", "collaboration"],
                "job_level": "research engineer",
                "duration_limit": 30
            }
        },
        {
            "query": "I am hiring for an analyst and want applications to screen using Cognitive and personality tests, what options are available within 45 mins",
            "expected_filters": {
                "skills": ["cognitive", "personality"],
                "job_level": "analyst",
                "duration_limit": 45
            }
        }
    ]

    results = []
    passed = 0
    
    for i, case in enumerate(test_cases):
        query = case["query"]
        expected = case["expected_filters"]
        
        # Process the recommendation without depending on LangSmith
        filters = parse_query_with_gemini(query)
        recommendations = recommend_assessments(filters)
        
        # Log to LangSmith if available (optional)
        run_id = None
        if use_langsmith:
            try:
                # Create a simple run to log results
                run = client.create_run(
                    project_name="shl-recommender",
                    name=f"Evaluation run for query {i}",
                    inputs={"query": query},
                    outputs={"filters": filters, "recommendations": [r["title"] for r in recommendations[:3]]},
                    tags=["evaluation"]
                )
                run_id = run.id
            except Exception as e:
                print(f"Warning: Could not log to LangSmith: {e}")
        
        # Standard evaluation criteria
        extracted_skills = [s.lower() for s in filters.get("skills", [])] if filters.get("skills") else []
        extracted_job_level = filters.get("job_level", "").lower() if filters.get("job_level") else None
        
        skills_pass = all(any(expected_skill in extracted_skill for extracted_skill in extracted_skills) 
                        for expected_skill in expected["skills"])
        
        job_level_pass = expected["job_level"] is None or (
            extracted_job_level and expected["job_level"].lower() in extracted_job_level
        )
        
        duration_pass = expected["duration_limit"] is None or (
            filters.get("duration_limit") is not None and 
            abs(filters.get("duration_limit") - expected["duration_limit"]) <= 5
        )
        
        case_passed = skills_pass and duration_pass and job_level_pass
        
        # Save detailed results
        result = {
            "query": query,
            "expected": expected,
            "extracted": filters,
            "passed": case_passed,
            "checks": {
                "skills_pass": skills_pass,
                "job_level_pass": job_level_pass,
                "duration_pass": duration_pass
            },
            "top_recommendations": [r["title"] for r in recommendations[:3]],
            "langsmith_run_id": run_id
        }
        
        results.append(result)
        
        if case_passed:
            print(f"✅ Passed for: {query}")
            passed += 1
        else:
            print(f"❌ Failed for: {query}")
            print("   ↪ Extracted:", filters)
            print("   ↪ Expected :", expected)
            
            # Debug which checks failed
            if not skills_pass:
                print("   ↪ Skills check failed")
            if not job_level_pass:
                print("   ↪ Job level check failed")
            if not duration_pass:
                print("   ↪ Duration check failed")

    print(f"\n📊 Eval summary: {passed}/{len(test_cases)} passed.")
    
    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📝 Evaluation results saved to {output_file}")
    if use_langsmith:
        print(f"🔗 View detailed runs in LangSmith: https://smith.langchain.com/projects/shl-recommender")


@app.command()
def view_recent_traces(limit: int = 10):
    """View recent traces from LangSmith"""
    from langsmith import Client
    from tabulate import tabulate
    from datetime import datetime
    
    client = Client()
    project_name = "shl-recommender"
    
    print(f"🔍 Fetching {limit} most recent traces from project: {project_name}")
    
    # Get recent runs
    try:
        runs = client.list_runs(
            project_name=project_name,
            limit = int(limit)
        )
        
        table_data = []
        for run in runs:
            # Format timestamp
            timestamp = datetime.fromisoformat(run.start_time.replace('Z', '+00:00'))
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract query if available
            query = run.inputs.get("query", "N/A") if run.inputs else "N/A"
            
            # Truncate long queries
            if len(query) > 50:
                query = query[:47] + "..."
            
            table_data.append([
                run.id[:8] + "...",  # Truncated ID
                run.name,
                formatted_time,
                query,
                run.error or "Success"
            ])
        
        # Print table
        print(tabulate(
            table_data,
            headers=["Run ID", "Name", "Timestamp", "Query", "Status"],
            tablefmt="grid"
        ))
        
        print(f"\n🔗 View all runs: https://smith.langchain.com/projects/{project_name}")
    
    except Exception as e:
        print(f"❌ Error fetching traces: {e}")
        print("💡 Make sure your LANGCHAIN_API_KEY is set correctly")

@app.command()
def get_dashboard_link():
    """Get a link to your LangSmith dashboard for this project"""
    from langsmith import Client
    
    client = Client()
    project_name = "shl-recommender"
    
    # Verify project exists
    projects = client.list_projects()
    project_exists = any(p.name == project_name for p in projects)
    
    if project_exists:
        print(f"🔗 LangSmith Dashboard: https://smith.langchain.com/projects/{project_name}")
    else:
        print(f"⚠️ Project '{project_name}' not found. Have you run any traces yet?")
        print("💡 Try running a recommendation or evaluation first.")

if __name__ == "__main__":
    print("Starting CLI application...")
    print(f"Command line arguments: {sys.argv}")
    try:
        app()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    print("CLI application finished.")