def build_extraction_prompt_old(query: str) -> str:
    return f"""
You are an AI assistant that extracts structured hiring criteria from user prompts.

Given the following query:
\"\"\"{query}\"\"\"

Extract and return a JSON with:
- skills: list of programming/technical skills
- soft_skills: list of soft skills (e.g., communication, collaboration)
- job_level: junior, mid, or senior (guess if not given)
- duration_limit: number (max time in minutes for assessment)

Return only the JSON. No explanation.
"""

def build_extraction_prompt(query: str) -> str:
        return f"""
You are an intelligent assistant. Your task is to extract structured fields from the following user query.

Extract ONLY the following fields:
- "skills": list of strings (e.g., ["Python", "Machine Learning"])
- "job_level": string (e.g., "Entry", "Mid", "Senior")
- "duration_limit": integer (duration in minutes)

Rules:
- Be concise. Do not include extra explanation.
- If the query mentions specific fields like "Generative AI", infer parent skills like "AI".
- If the job level is unclear, leave it as null.
- Always respond with a valid JSON object only.
- Convert all skill names to their standard forms (e.g., "Java Script" should be "JavaScript", "ML" should be "Machine Learning")
- Pay special attention to job roles mentioned (e.g., "research engineer", "developer", "analyst") and include them in job_level
- Only extract the core skills without adding words like "assessment" or "testing" (e.g., use "Cognitive" not "Cognitive assessment")
- Look for collaboration skills when mentioned in a teamwork context
- Convert all values to lowercase for consistency

Query: "{query}"
"""
