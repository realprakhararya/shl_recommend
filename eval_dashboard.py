from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
from pathlib import Path
import uvicorn
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

app = FastAPI()

# Create templates directory if it doesn't exist
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

# Create a basic HTML template
template_path = templates_dir / "dashboard.html"
if not template_path.exists():
    template_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Assessment Recommender Evaluation</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .passed { background-color: #d4edda; }
            .failed { background-color: #f8d7da; }
            .container { max-width: 1200px; }
            .chart-container { height: 400px; }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h1>Assessment Recommender Evaluation Dashboard</h1>
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Evaluation Summary</h5>
                        </div>
                        <div class="card-body">
                            <p>Run Date: {{ run_date }}</p>
                            <p>Total Cases: {{ total_cases }}</p>
                            <p>Passed: {{ passed_cases }} ({{ pass_rate }}%)</p>
                            <div class="chart-container">
                                <canvas id="summaryChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Check Types Breakdown</h5>
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="checksChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card mt-4">
                <div class="card-header">
                    <h5>Test Cases</h5>
                </div>
                <div class="card-body">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Query</th>
                                <th>Expected</th>
                                <th>Extracted</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for result in results %}
                            <tr class="{{ 'passed' if result.passed else 'failed' }}">
                                <td>{{ loop.index }}</td>
                                <td>{{ result.query }}</td>
                                <td>{{ result.expected | tojson }}</td>
                                <td>{{ result.extracted | tojson }}</td>
                                <td>
                                    {% if result.passed %}
                                    <span class="badge bg-success">PASSED</span>
                                    {% else %}
                                    <span class="badge bg-danger">FAILED</span>
                                    {% endif %}
                                </td>
                                <td>
                                    {% if result.langsmith_run_id %}
                                    <a href="https://smith.langchain.com/runs/{{ result.langsmith_run_id }}" 
                                       class="btn btn-sm btn-primary" target="_blank">
                                        View in LangSmith
                                    </a>
                                    {% endif %}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <script>
            // Summary chart
            const summaryCtx = document.getElementById('summaryChart').getContext('2d');
            const summaryChart = new Chart(summaryCtx, {
                type: 'pie',
                data: {
                    labels: ['Passed', 'Failed'],
                    datasets: [{
                        data: [{{ passed_cases }}, {{ total_cases - passed_cases }}],
                        backgroundColor: ['#28a745', '#dc3545']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
            
            // Checks breakdown chart
            const checksCtx = document.getElementById('checksChart').getContext('2d');
            const checksChart = new Chart(checksCtx, {
                type: 'bar',
                data: {
                    labels: ['Skills', 'Job Level', 'Duration'],
                    datasets: [{
                        label: 'Passed',
                        data: [{{ check_stats.skills_pass }}, {{ check_stats.job_level_pass }}, {{ check_stats.duration_pass }}],
                        backgroundColor: '#28a745'
                    }, {
                        label: 'Failed',
                        data: [
                            {{ total_cases - check_stats.skills_pass }}, 
                            {{ total_cases - check_stats.job_level_pass }}, 
                            {{ total_cases - check_stats.duration_pass }}
                        ],
                        backgroundColor: '#dc3545'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            stacked: true
                        },
                        y: {
                            stacked: true,
                            beginAtZero: true
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    with open(template_path, "w") as f:
        f.write(template_content)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the evaluation dashboard"""
    # Default to the latest eval file
    eval_files = sorted(Path(".").glob("eval_results*.json"), reverse=True)
    
    if not eval_files:
        return HTMLResponse(content="<h1>No evaluation results found</h1>")
    
    eval_file = eval_files[0]
    
    # Load the evaluation results
    with open(eval_file, "r") as f:
        results = json.load(f)
    
    # Calculate statistics
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r["passed"])
    pass_rate = round((passed_cases / total_cases) * 100, 1) if total_cases > 0 else 0
    
    # Check type statistics
    check_stats = {
        "skills_pass": sum(1 for r in results if r["checks"]["skills_pass"]),
        "job_level_pass": sum(1 for r in results if r["checks"]["job_level_pass"]),
        "duration_pass": sum(1 for r in results if r["checks"]["duration_pass"])
    }
    
    # Get run date from file modification time
    run_date = datetime.fromtimestamp(eval_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": request,
            "results": results,
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "pass_rate": pass_rate,
            "check_stats": check_stats,
            "run_date": run_date
        }
    )

@app.get("/api/results")
async def get_results():
    """API endpoint to get the evaluation results"""
    eval_files = sorted(Path(".").glob("eval_results*.json"), reverse=True)
    
    if not eval_files:
        return {"error": "No evaluation results found"}
    
    with open(eval_files[0], "r") as f:
        results = json.load(f)
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)