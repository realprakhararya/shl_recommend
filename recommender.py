import pandas as pd
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from prompts import build_extraction_prompt
import re
from config import SCORING_WEIGHTS, get_model
from langsmith import traceable

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load and clean SHL dataset
shl_df = pd.read_csv("shl_clean.csv", encoding='latin1')
shl_df = shl_df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))

if "topic" in shl_df.columns:
    shl_df = shl_df.rename(columns={"topic": "title"})

if "assessment_length" in shl_df.columns:
    # Keep original values, don't fill with 60
    shl_df["assessment_length"] = pd.to_numeric(shl_df["assessment_length"], errors="coerce")
    # Only fill NaN values with 0 instead of 60
    shl_df["assessment_length"] = shl_df["assessment_length"].fillna(0)

def exact_skill_match_count(row, skills):
    text = f"{row['title']} {row['description']}".lower()
    return sum(1 for skill in skills if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text))

def parse_query_with_gemini_old(query: str) -> dict:
    prompt = build_extraction_prompt(query)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    print("Gemini response:\n", response.text)

    try:
        # Remove Markdown code block if present
        cleaned = response.text.strip().strip("```").replace("json", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        print(f"Gemini parsing failed: {e}")
        return {}
    
#@traceable(name="parse_query_with_gemini")
def parse_query_with_gemini(query: str) -> dict:
    prompt = build_extraction_prompt(query)
    print("Gemini prompt:\n", prompt)

    model = get_model()
    response = model.generate_content(prompt)

    print("\nGemini response:\n", response.text)
    
    # Extract JSON from response (it might be wrapped in markdown code blocks)
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response.text)
    
    if match:
        json_str = match.group(1)
    else:
        json_str = response.text
    
    # Clean up any remaining non-JSON content
    json_str = json_str.strip()
    print("\n🧪 Cleaned JSON string:\n", json_str)
    
    try:
        filters = json.loads(json_str)
        
        # Convert all lists and strings to lowercase
        if "skills" in filters and filters["skills"]:
            filters["skills"] = [skill.lower() for skill in filters["skills"]]
            
        if "job_level" in filters and filters["job_level"]:
            filters["job_level"] = filters["job_level"].lower()
            
        return filters
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {"skills": [], "job_level": None, "duration_limit": None}

def apply_scores(df):
    df['score'] = (
        df['technical_score'] * SCORING_WEIGHTS['technical_score'] +
        df['title_relevance'] * SCORING_WEIGHTS['title_relevance'] +
        df['inferred_score'] * SCORING_WEIGHTS['inferred_score'] +
        df['soft_skill_score'] * SCORING_WEIGHTS['soft_skill_score']
    )
    return df

def recommend(df, top_k=5):
    df = apply_scores(df)
    df = df.sort_values(by='score', ascending=False)
    return df.head(top_k)

def tokenize_term(term):
    """Break a term into its component words and parts"""
    # Convert to lowercase
    term = term.lower()
    
    # Replace hyphens and underscores with spaces
    term = term.replace('-', ' ').replace('_', ' ')
    
    # Split into words
    words = term.split()
    
    # Add the original term and individual words
    tokens = [term] + words
    
    # Add combinations of adjacent words
    for i in range(len(words) - 1):
        tokens.append(words[i] + ' ' + words[i+1])
    
    return set(tokens)

def terms_overlap(term1, term2):
    """Check if two terms overlap by breaking them into tokens"""
    tokens1 = tokenize_term(term1)
    tokens2 = tokenize_term(term2)
    
    # Check for any overlap between token sets
    return bool(tokens1.intersection(tokens2))

#@traceable(name="recommend_assessments")
def recommend_assessments(filters: dict):

    df = shl_df.copy()
    
    # Initialize scoring columns for all possible paths
    df['technical_score'] = 0
    df['soft_skill_score'] = 0
    df['inferred_score'] = 0
    df['job_level_score'] = 0
    df['description_match_score'] = 0
    df['title_relevance'] = 0

    # --- Skill filtering - But don't filter, just score ---
    if 'skills' in filters and filters['skills']:
        # Identify soft skills
        soft_skills = {'collaboration', 'communication', 'teamwork', 'leadership', 
                     'problem solving', 'critical thinking', 'adaptability', 
                     'time management', 'interpersonal', 'work ethic', 'creativity',
                     'organization', 'attention to detail', 'management', 'negotiation'}
        
        # Separate technical and soft skills
        technical_skills = []
        identified_soft_skills = []
        
        for skill in filters['skills']:
            is_soft = False
            for soft_skill in soft_skills:
                if soft_skill.lower() in skill.lower():
                    identified_soft_skills.append(skill)
                    is_soft = True
                    break
            if not is_soft:
                technical_skills.append(skill)
        
        # Expand known skills into supersets
        superset_map = {
            'generative ai': ['ai', 'machine learning', 'deep learning', 'llm', 'large language models'],
            'llm': ['ai', 'machine learning', 'nlp', 'natural language processing'],
            'nlp': ['ai', 'machine learning', 'natural language processing'],
            'computer vision': ['ai', 'ml', 'machine learning', 'deep learning', 'image processing'],
            'chatgpt': ['generative ai', 'llm', 'ai', 'language model', 'nlp'],
            'data science': ['statistics', 'analytics', 'data analysis', 'ml'],
            'frontend': ['javascript', 'html', 'css', 'web development', 'ui'],
            'backend': ['api', 'server', 'database', 'web development'],
            'machine learning': ['ai', 'algorithms', 'data science'],
            'deep learning': ['ai', 'neural networks', 'machine learning'],
            'devops': ['ci/cd', 'cloud', 'infrastructure', 'deployment'],
            'cloud computing': ['aws', 'azure', 'gcp', 'infrastructure']
        }
        
        expanded_tech_skills = set()
        for skill in technical_skills:
            expanded_tech_skills.add(skill)
            for super_key, supers in superset_map.items():
                if skill.lower() in super_key or super_key in skill.lower():
                    expanded_tech_skills.update(supers)
        
        technical_skills = list(expanded_tech_skills)
        
        print(f"Technical skills (expanded): {technical_skills}")
        print(f"Soft skills: {identified_soft_skills}")

        # Function to infer skills from role titles
        def infer_skills_from_role(title):
            title = str(title).lower()
            inferred = []
            
            role_skill_map = {
                'research': ['ai', 'ml', 'data science', 'analytics', 'algorithms'],
                'engineer': ['software development', 'programming', 'technical'],
                'research engineer': ['ai', 'ml', 'machine learning', 'data science', 'algorithms'],
                'data scientist': ['data science', 'statistics', 'machine learning', 'python'],
                'developer': ['programming', 'software development', 'coding'],
                'analyst': ['data analysis', 'analytics', 'statistics'],
                'designer': ['ui', 'ux', 'design', 'creative'],
                'manager': ['leadership', 'management', 'team', 'project management'],
                'product': ['product management', 'strategy', 'roadmap']
            }
            
            for role, skills in role_skill_map.items():
                if role in title:
                    inferred.extend(skills)
            
            return inferred
        
        # SCORE technical skills matches (but don't filter)
        if technical_skills:
            # Calculate scores even for entries that don't have exact matches
            df['technical_score'] = df.apply(lambda row: exact_skill_match_count(row, technical_skills), axis=1)
            
            # Add title relevance score for more relevant weighting
            df['title_relevance'] = df.apply(
                lambda row: sum(
                    str(row['title']).lower().count(skill.lower()) * 2  # Title matches worth double
                    for skill in technical_skills
                ), 
                axis=1
            )
            
            # Add inferred skills score
            df['inferred_skills'] = df['title'].apply(infer_skills_from_role)
            df['inferred_score'] = df.apply(
                lambda row: sum(
                    1 for skill in row['inferred_skills'] 
                    if not any(tech.lower() in skill.lower() or skill.lower() in tech.lower() for tech in technical_skills)
                ), 
                axis=1
            )
            
        # Score soft skills as bonus
        if identified_soft_skills:
            df['soft_skill_score'] = df.apply(
                lambda row: sum(
                    skill.lower() in f"{str(row['title'])} {str(row['description'])}".lower()
                    for skill in identified_soft_skills
                ),
                axis=1
            )
            
        # Calculate score from combination of technical, inferred, and soft skills
        # Technical skills have more weight than soft skills
        df['score'] = apply_scores(df)['score']
        
        # Add description relevance score
        all_skills = technical_skills + identified_soft_skills
        if all_skills:
            # Calculate how much of the description matches any skill
            df['description_match_score'] = df.apply(
                lambda row: sum(
                    str(row['description']).lower().count(skill.lower()) 
                    for skill in all_skills
                ) / (len(str(row['description'])) + 1) * 100,  # Normalize by description length
                axis=1
            )
    else:
        df['score'] = 0
        df['description_match_score'] = 0

    # --- Job level scoring (not filtering) ---
    job_level = filters.get('job_level')
    if job_level and isinstance(job_level, str) and 'job_levels' in df.columns:
        # Base job level categories
        job_level_mapping = {
            'entry': ['entry', 'junior', 'beginner', 'novice', 'entry level', 'entry-level'],
            'mid': ['mid', 'intermediate', 'middle', 'mid-level', 'mid level', 'mid-career', 'mid professional'],
            'senior': ['senior', 'advanced', 'expert', 'lead', 'senior level', 'senior-level'],
            'executive': ['executive', 'c-level', 'director', 'manager', 'management']
        }
        
        # Infer job level from role name
        role_level_mapping = {
            'research engineer': ['mid', 'senior'],
            'senior': ['senior'],
            'lead': ['senior'],
            'principal': ['senior'],
            'director': ['executive'],
            'manager': ['executive'],
            'head': ['executive'],
            'chief': ['executive'],
            'junior': ['entry'],
            'associate': ['entry', 'mid'],
            'intern': ['entry']
        }
        
        # Find category using token overlap
        matched_category = None
        for category, terms in job_level_mapping.items():
            if any(terms_overlap(job_level, term) for term in terms):
                matched_category = category
                break
                
        # Apply job level scoring based on matched category (but don't filter)
        if 'job_levels' in df.columns:
            if matched_category:
                # Get all terms from the matched category for pattern matching
                category_terms = job_level_mapping[matched_category]
                print(f"Job level '{job_level}' matched to category: {matched_category}")
                print(f"Using terms for scoring: {category_terms}")
                
                # Score based on matched category
                df['job_level_score'] = df.apply(
                    lambda row: 2 if any(
                        term.lower() in str(row['job_levels']).lower() 
                        for term in category_terms
                    ) else 0,
                    axis=1
                )
                
                # Also try to match from job title for roles
                for role, levels in role_level_mapping.items():
                    if matched_category in levels:
                        df['job_level_score'] += df.apply(
                            lambda row: 3 if role.lower() in str(row['title']).lower() else 0,
                            axis=1
                        )
            else:
                # Try to infer from the job title itself
                print(f"No direct category match for '{job_level}', inferring from title/description")
                
                tokens = tokenize_term(job_level)
                # Only use tokens that are meaningful (3+ characters)
                meaningful_tokens = [t for t in tokens if len(t) >= 3]
                
                df['job_level_score'] = df.apply(
                    lambda row: sum(
                        2 for token in meaningful_tokens 
                        if token in str(row['title']).lower() or token in str(row['job_levels']).lower()
                    ),
                    axis=1
                )
            
            # Add job level score to total score
            df['score'] += df['job_level_score']

    # --- Duration limit filtering (only if strict match available) ---
    duration_limit = filters.get('duration_limit')
    if duration_limit is not None:
        try:
            duration_limit = float(duration_limit)
            
            # More relaxed filtering approach - add as a score factor first
            df['duration_match'] = df.apply(
                lambda row: 3 if row['assessment_length'] <= duration_limit else 
                           (1 if row['assessment_length'] <= duration_limit*1.5 else 0),
                axis=1
            )
            
            df['score'] += df['duration_match']
            
            # Only filter if we'd still have results
            filtered_df = df[df['assessment_length'] <= duration_limit]
            if len(filtered_df) >= 3:  # Make sure we have at least 3 results
                df = filtered_df
            else:
                print(f"Keeping all results despite duration limit {duration_limit}, added as score factor instead")
                
        except ValueError:
            pass

    # --- Final cleanup ---
    if df.empty:
        return []

    # Select relevant columns
    result_df = df[[
        'title', 'description', 'job_levels', 'language',
        'assessment_length', 'test_type', 'remote_testing',
        'adaptive/irt', 'score', 'description_match_score'
    ]]

    # Always return results, sorted by score
    result_df = result_df.fillna("N/A").sort_values(by=['score', 'description_match_score'], ascending=[False, False]).head(10)
    
    # Drop the scoring columns from the final output
    result_df = result_df.drop(columns=['description_match_score'])

    return result_df.to_dict(orient='records')
