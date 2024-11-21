import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import openai
import numpy as np
import os
from scipy.stats import pearsonr

# ------------------------------
# Initialize Legal-BERT tokenizer and model
# ------------------------------
# Legal-BERT is a pretrained model specialized for legal text embeddings.
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Clause database: A collection of legal clauses for comparison
clause_database = [
    """
    This Agreement is made and entered into on this 13th day of November, 2024, by and between XYZ Baseball Club,
    a professional baseball team with its principal office located at 789 Stadium Lane, Chicago, IL ("Team"),
    and John Smith, an individual residing at 123 Athlete Drive, Los Angeles, CA ("Player").
    """,
    """
    1. Term of Employment. The Team agrees to employ the Player, and the Player agrees to be employed
    by the Team as a professional baseball player for the term beginning on February 1, 2025, and ending
    on October 31, 2025, unless terminated earlier in accordance with this Agreement.
    """,
    """
    2. Compensation. The Team agrees to pay the Player a total salary of $1,500,000 for the duration
    of the term, payable in equal installments in accordance with the Team's regular payroll schedule.
    """,
    """
    3. Performance Bonuses. In addition to the base salary, the Player shall be eligible for the following
    performance bonuses:
        (a) $25,000 for every home run exceeding 20 home runs during the regular season.
        (b) $50,000 if the Player is selected for the All-Star Game.
        (c) $100,000 if the Player wins the league MVP award.
    """,
    """
    4. Duties and Responsibilities. The Player agrees to perform to the best of their ability as a professional
    baseball player. This includes attending all games, practices, training sessions, promotional events,
    and any other activities reasonably required by the Team.
    """,
    """
    5. Medical Examination. The Player agrees to undergo a medical examination by a physician designated
    by the Team prior to the commencement of employment. The Player's employment is contingent upon
    passing such medical examination.
    """,
    """
    6. Termination. The Team may terminate this Agreement prior to its expiration under the following
    circumstances:
        (a) If the Player is unable to perform their duties for a period exceeding 60 consecutive days
        due to injury or illness.
        (b) For cause, including but not limited to violation of league rules, conduct detrimental to
        the Team, or breach of this Agreement.
    """,
    """
    7. Confidentiality. The Player agrees to maintain the confidentiality of any proprietary or confidential
    information disclosed by the Team, including but not limited to playbooks, strategies, and training
    techniques.
    """,
    """
    8. Governing Law. This Agreement shall be governed by and construed in accordance with the laws
    of the State of Illinois.
    """,
    """
    9. Entire Agreement. This Agreement constitutes the entire agreement between the Parties with respect
    to the subject matter hereof and supersedes all prior agreements, understandings, or representations,
    whether written or oral.
    """
]

# ------------------------------
# Helper Functions
# ------------------------------

def get_embedding(text):
    """
    Generate embeddings for a given text using Legal-BERT.

    Args:
        text (str): The text to encode.

    Returns:
        torch.Tensor: The CLS token embedding.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # Extract CLS token embedding
    return embedding

# Precompute embeddings for the clause database
clause_embeddings = torch.cat([get_embedding(clause) for clause in clause_database], dim=0)

def find_similar_clauses(query, clause_database, clause_embeddings, top_n=3):
    """
    Find clauses similar to a query using cosine similarity.

    Args:
        query (str): The query clause.
        clause_database (list): List of clauses.
        clause_embeddings (torch.Tensor): Precomputed embeddings of the database.
        top_n (int): Number of top results to return.

    Returns:
        tuple: List of similar clauses and their similarity scores.
    """
    query_embedding = get_embedding(query)
    similarities = cosine_similarity(query_embedding, clause_embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    similar_clauses = [clause_database[i] for i in top_indices]
    similarity_scores = similarities[0][top_indices]
    return similar_clauses, similarity_scores

# ------------------------------
# OpenAI GPT Integration
# ------------------------------

# Initialize OpenAI client
client = openai.Client()

def analyze_risks_with_context(query_clause, similar_clauses, openai_key):
    """
    Perform risk analysis on a clause with contextual clauses using OpenAI GPT.

    Args:
        query_clause (str): The main clause to analyze.
        similar_clauses (list): Contextual clauses for analysis.
        openai_key (str): OpenAI API key.

    Returns:
        str: Risk analysis generated by GPT.
    """
    openai.api_key = openai_key

    # Combine the query clause with context
    context = "\n\n".join(f"Relevant Clause {i+1}: {clause}" for i, clause in enumerate(similar_clauses))
    risk_template = """
    You are a legal advisor. Below is a clause that needs risk analysis, along with relevant clauses from other agreements for context.
    Please identify any potential risks in the primary clause and explain how the context might affect your analysis.
    """
    content = f"{risk_template}\n\nPrimary Clause:\n{query_clause}\n\nContext from Similar Clauses:\n{context}"

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="gpt-4o",
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_risks_without_context(query_clause, openai_key):
    """
    Perform risk analysis on a clause without contextual clauses using OpenAI GPT.

    Args:
        query_clause (str): The main clause to analyze.
        openai_key (str): OpenAI API key.

    Returns:
        str: Risk analysis generated by GPT.
    """
    openai.api_key = openai_key
    risk_template = """
    You are a legal advisor. Below is a clause that needs risk analysis.
    Please identify any potential risks in the clause and provide detailed explanations.
    """
    content = f"{risk_template}\n\nPrimary Clause:\n{query_clause}"

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="gpt-4o",
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------------
# Embedding Similarity Comparison
# ------------------------------

# Load a sentence embedding model for output comparison
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

def generate_embedding(text):
    """
    Generate sentence embeddings using a sentence-transformer model.

    Args:
        text (str): The text to encode.

    Returns:
        torch.Tensor: Sentence embedding.
    """
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def compute_pearson_correlation(output1, output2):
    """
    Compute Pearson correlation between two text outputs.

    Args:
        output1 (str): First output.
        output2 (str): Second output.

    Returns:
        float: Pearson correlation coefficient.
    """
    embedding1 = generate_embedding(output1).numpy()
    embedding2 = generate_embedding(output2).numpy()
    correlation, _ = pearsonr(embedding1, embedding2)
    return correlation

# ------------------------------
# Main Execution
# ------------------------------

# Example query clause
query_clause = "Discuss confidentiality agreements and related risks."
similar_clauses, scores = find_similar_clauses(query_clause, clause_database, clause_embeddings)

# OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")

if openai_key:
    risk_analysis1 = analyze_risks_with_context(query_clause, similar_clauses, openai_key)
    risk_analysis2 = analyze_risks_without_context(query_clause, openai_key)
    
    # Compute Pearson correlation between the outputs
    pearson_corr = compute_pearson_correlation(risk_analysis1, risk_analysis2)
    print("Risk Analysis with Context:")
    print(risk_analysis1)
    print("\nRisk Analysis Without Context:")
    print(risk_analysis2)
    print(f"\nPearson Correlation between outputs: {pearson_corr:.4f}")
else:
    print("Please set the OPENAI_API_KEY environment variable.")
