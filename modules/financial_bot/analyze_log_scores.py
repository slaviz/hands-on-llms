import os
import re
from collections import defaultdict

def parse_scores(log_file):
    """
    Parse scores from the log file
    """
    score_pattern = r"Score={'ragas_score': (\d+\.\d+), 'context_relevancy': (\d+\.\d+), 'context_recall': (\d+\.\d+), 'answer_similarity': (\d+\.\d+), 'faithfulness': (\d+\.\d+)}"
    scores = defaultdict(list)
    
    try:
        with open(log_file, 'r') as file:
            for line in file:
                match = re.search(score_pattern, line)
                if match:
                    ragas_score, context_relevancy, context_recall, answer_similarity, faithfulness = map(float, match.groups())
                    scores['ragas_score'].append(ragas_score)
                    scores['context_relevancy'].append(context_relevancy)
                    scores['context_recall'].append(context_recall)
                    scores['answer_similarity'].append(answer_similarity)
                    scores['faithfulness'].append(faithfulness)
        
        return scores
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None

def calculate_averages(scores):
    """
    Calculate average scores
    """
    return {
        key: round(sum(values) / len(values), 4) if values else 0 
        for key, values in scores.items()
    }

def main():
    # Define log file path
    log_file = os.path.join('logs', 'info.log')
    
    # Check if the file exists
    if not os.path.exists(log_file):
        print(f"Error: The file {log_file} does not exist.")
        return
    
    # Parse scores from the log file
    scores = parse_scores(log_file)
    
    if not scores:
        print("No scores found in the log file.")
        return
    
    # Calculate averages
    averages = calculate_averages(scores)
    
    # Print results
    print(f"Analyzing log file: {log_file}")
    print("\nAverage Scores:")
    for key, value in averages.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Additional statistics
    print("\nNumber of score entries:")
    for key, value in scores.items():
        print(f"{key.replace('_', ' ').title()}: {len(value)} entries")

if __name__ == "__main__":
    main()
