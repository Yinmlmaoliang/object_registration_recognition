"""
Home Service Robot - Personalized Object Representation and Retrieval Evaluation
Final comprehensive evaluation (English Optimized)
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import time
import psutil
import os
from collections import defaultdict

# ========== CONFIGURATION ==========
# Using a model that works well for English semantic similarity
MODEL_NAME = 'all-MiniLM-L6-v2'  # Or keep 'paraphrase-multilingual-MiniLM-L12-v2'
ENCODING_STRATEGY = 'max'  # max pooling performs best
TOP_K_VALUES = [1, 3, 5]

# ========== INITIALIZE MODEL ==========
print(f"Loading model: {MODEL_NAME}")
start_time = time.time()
model = SentenceTransformer(MODEL_NAME)
load_time = time.time() - start_time
embedding_dim = model.get_sentence_embedding_dimension()
print(f"[SUCCESS] Model loaded: {MODEL_NAME} ({embedding_dim}D)")
print(f"          Load time: {load_time:.2f}s")
print("=" * 70)

# ========== OBJECT DATABASE ==========
# Designed for personalized home service:
# - No ambiguous visual features (color, location)
# - Focus on: Function, Ownership, Usage Habits, Special Properties
object_database = {
    "obj_001": {
        "name": "Personal Mug",
        "category": "Kitchenware",
        "attributes": [
            "my favorite mug",
            "the cup I use for morning coffee",
            "my daily water cup",
            "the mug with the chipped handle",
            "cup for hot beverages"
        ]
    },
    "obj_002": {
        "name": "Guest Mug",
        "category": "Kitchenware",
        "attributes": [
            "mug for visitors",
            "spare cup",
            "clean mug for guests",
            "cup I rarely use",
            "standard serving cup"
        ]
    },
    "obj_003": {
        "name": "Thermos Bottle",
        "category": "Kitchenware",
        "attributes": [
            "insulated bottle",
            "container that keeps water hot",
            "flask for hiking",
            "bottle for carrying tea",
            "vacuum flask"
        ]
    },
    "obj_004": {
        "name": "Noise Cancelling Headphones",
        "category": "Electronics",
        "attributes": [
            "headphones for deep work",
            "noise cancelling headset",
            "audio device for flights",
            "my expensive music gear",
            "headset for focus"
        ]
    },
    "obj_005": {
        "name": "Gaming Headset",
        "category": "Electronics",
        "attributes": [
            "headphones with a microphone",
            "headset for playing video games",
            "bulky headphones for PC",
            "audio gear for discord",
            "wired headset"
        ]
    },
    "obj_006": {
        "name": "House Keys",
        "category": "Personal",
        "attributes": [
            "keys to open the front door",
            "bunch of keys with a keychain",
            "important item to take when leaving",
            "metal keys for the apartment",
            "thing to unlock the door"
        ]
    },
    "obj_007": {
        "name": "Car Keys",
        "category": "Personal",
        "attributes": [
            "remote key for the vehicle",
            "key fob for driving",
            "keys to start the engine",
            "black electronic key",
            "thing to unlock the car"
        ]
    },
    "obj_008": {
        "name": "Daily Vitamins",
        "category": "Health",
        "attributes": [
            "pills I take after breakfast",
            "bottle of vitamin supplements",
            "health supplements",
            "daily medication container",
            "plastic jar with pills"
        ]
    },
    "obj_009": {
        "name": "First Aid Kit",
        "category": "Health",
        "attributes": [
            "emergency medical box",
            "kit with band-aids and antiseptic",
            "box for injuries",
            "medical supplies container",
            "emergency health kit"
        ]
    },
    "obj_010": {
        "name": "Work Notebook",
        "category": "Stationery",
        "attributes": [
            "journal for meeting notes",
            "notebook with important work ideas",
            "my professional diary",
            "pad for writing drafts",
            "book where I write tasks"
        ]
    },
    "obj_011": {
        "name": "Sketchbook",
        "category": "Stationery",
        "attributes": [
            "book for drawing",
            "paper pad for art",
            "sketching notebook",
            "my creative outlet",
            "book with blank pages"
        ]
    },
    "obj_012": {
        "name": "Dog's Toy",
        "category": "Pet",
        "attributes": [
            "chew toy for the pet",
            "squeaky rubber bone",
            "dog's favorite plaything",
            "toy for fetching",
            "item the dog plays with"
        ]
    }
}

# ========== ENCODING FUNCTION ==========
def encode_object_max(model, attributes):
    """Encode attributes using max pooling strategy"""
    embeddings = model.encode(attributes, convert_to_tensor=True)
    return embeddings.max(dim=0).values

# ========== LEARNING PHASE ==========
print(f"\n[LEARNING] Encoding object attributes using '{ENCODING_STRATEGY}' strategy...")
object_embeddings = {}
encoding_start = time.time()

for obj_id, obj_info in object_database.items():
    embedding = encode_object_max(model, obj_info["attributes"])
    object_embeddings[obj_id] = {
        "name": obj_info["name"],
        "category": obj_info["category"],
        "embedding": embedding,
        "attributes": obj_info["attributes"]
    }
    print(f"  [OK] {obj_id}: {obj_info['name']}")

encoding_time = time.time() - encoding_start
print(f"\n          Encoding time: {encoding_time:.3f}s ({len(object_database)} objects)")
print(f"          Average per object: {encoding_time/len(object_database)*1000:.2f}ms")

# Memory usage
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"          Memory usage: {memory_mb:.1f}MB")
print("=" * 70)

# ========== TEST QUERIES ==========
test_queries = [
    # 1. Specific Usage / Habits
    ("I need my morning coffee cup", "obj_001", "Specific Habit"),
    ("Get the mug I use every day", "obj_001", "Frequency"),
    ("Bring me the bottle that keeps water hot", "obj_003", "Function Specific"),
    
    # 2. Social / Occasion
    ("We have a visitor, get a clean cup", "obj_002", "Social Context"),
    ("I need a spare mug", "obj_002", "Status"),
    
    # 3. Work vs Play (Differentiation)
    ("I need to focus on my work, get my headset", "obj_004", "Context: Work"),
    ("I want to play some games, where is my headset?", "obj_005", "Context: Gaming"),
    ("I need to write down some meeting notes", "obj_010", "Context: Work"),
    ("I feel like drawing something", "obj_011", "Context: Creative"),

    # 4. Essential Items / Security
    ("I'm leaving the house, I need to lock the door", "obj_006", "Task: Leaving"),
    ("I need to drive to the supermarket", "obj_007", "Task: Driving"),
    
    # 5. Health / Emergency
    ("I need to take my daily supplements", "obj_008", "Routine: Health"),
    ("I cut my finger, find the medical box", "obj_009", "Emergency"),
    
    # 6. Pet
    ("The dog wants to play", "obj_012", "Agent: Pet"),

    # 7. Implicit / Vague
    ("My favorite drinking vessel", "obj_001", "Preference"),
    ("The thing for making loud noises quiet", "obj_004", "Functional Description"),
    ("The book with my drawings", "obj_011", "Content Description"),
    ("The electronic key", "obj_007", "Material/Type"),
]

# ========== METRICS CALCULATION ==========
def calculate_mrr(results_list):
    """Calculate Mean Reciprocal Rank"""
    return np.mean([1.0 / r['rank'] for r in results_list])

def calculate_top_k_accuracy(results_list, k):
    """Calculate Top-K accuracy"""
    return sum(1 for r in results_list if r['rank'] <= k) / len(results_list)

# ========== EXECUTE TESTS ==========
print("\n[RETRIEVAL] Testing query retrieval...")
print("=" * 70)

results = []
query_latencies = []
all_results_data = []

for query_text, expected_obj, query_type in test_queries:
    query_start = time.time()

    # Encode query
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    # Calculate similarities
    similarities = {}
    for obj_id, obj_data in object_embeddings.items():
        sim = util.cos_sim(query_embedding, obj_data["embedding"]).item()
        similarities[obj_id] = sim

    # Rank results
    ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top1_obj = ranked[0][0]
    top1_score = ranked[0][1]
    top2_score = ranked[1][1] if len(ranked) > 1 else 0

    query_latency = (time.time() - query_start) * 1000

    # Calculate rank
    rank = next((i+1 for i, (obj, _) in enumerate(ranked) if obj == expected_obj), len(ranked)+1)
    is_correct = (top1_obj == expected_obj)

    # Store results
    result_data = {
        'query': query_text,
        'query_type': query_type,
        'expected_obj': expected_obj,
        'expected_name': object_embeddings[expected_obj]["name"],
        'top1_obj': top1_obj,
        'top1_name': object_embeddings[top1_obj]["name"],
        'top1_score': top1_score,
        'top2_score': top2_score,
        'rank': rank,
        'is_correct': is_correct,
        'latency_ms': query_latency,
        'top3': [(obj, score, object_embeddings[obj]["name"]) for obj, score in ranked[:3]]
    }
    all_results_data.append(result_data)
    query_latencies.append(query_latency)

    # Print result
    status = "[PASS]" if is_correct else "[FAIL]"
    print(f"{status} [{query_type:18s}] {query_text[:40]:40s}")
    print(f"        Expected: {object_embeddings[expected_obj]['name']:20s} | Got: {object_embeddings[top1_obj]['name']:20s} (Rank #{rank}, Score: {top1_score:.4f})")
    top3_str = ' > '.join([f'{name}({score:.3f})' for _, score, name in result_data['top3']])
    print(f"        Top3: {top3_str}")
    print()

# ========== PERFORMANCE STATISTICS ==========
print("=" * 70)
print("\n[STATISTICS] Comprehensive Performance Analysis:")

# 1. Top-K Accuracy
print("\n1. Top-K Accuracy:")
for k in TOP_K_VALUES:
    acc = calculate_top_k_accuracy(all_results_data, k)
    hits = sum(1 for r in all_results_data if r['rank'] <= k)
    print(f"   Top-{k}: {acc*100:.1f}% ({hits}/{len(all_results_data)})")

# 2. MRR
mrr = calculate_mrr(all_results_data)
print(f"\n2. Mean Reciprocal Rank (MRR): {mrr:.4f}")

# 3. Performance by Query Type
print("\n3. Performance by Query Type:")
type_performance = defaultdict(list)
for result in all_results_data:
    type_performance[result['query_type']].append(result)

for qtype, results_list in sorted(type_performance.items()):
    acc = sum(1 for r in results_list if r['is_correct']) / len(results_list)
    mrr_type = calculate_mrr(results_list)
    avg_rank = np.mean([r['rank'] for r in results_list])
    print(f"   {qtype:18s}: Acc={acc*100:5.1f}% | MRR={mrr_type:.3f} | AvgRank={avg_rank:.1f}")

# 4. Similarity Score Analysis
print("\n4. Similarity Score Analysis:")
correct_scores = [r['top1_score'] for r in all_results_data if r['is_correct']]
incorrect_scores = [r['top1_score'] for r in all_results_data if not r['is_correct']]

if correct_scores:
    print(f"   Correct matches - Mean: {np.mean(correct_scores):.4f}, Min: {np.min(correct_scores):.4f}, Max: {np.max(correct_scores):.4f}")
if incorrect_scores:
    print(f"   Incorrect matches - Mean: {np.mean(incorrect_scores):.4f}, Min: {np.min(incorrect_scores):.4f}, Max: {np.max(incorrect_scores):.4f}")

# 5. Confidence Analysis
print("\n5. Confidence Analysis (Top1 - Top2 score gap):")
all_gaps = [r['top1_score'] - r['top2_score'] for r in all_results_data]
print(f"   Overall mean gap: {np.mean(all_gaps):.4f} +/- {np.std(all_gaps):.4f}")

# 6. Operational Performance
print("\n6. Operational Performance:")
print(f"   Mean query latency: {np.mean(query_latencies):.2f}ms")
print(f"   Memory usage: {memory_mb:.1f}MB")

# ========== FAILED CASES ANALYSIS ==========
failed_cases = [r for r in all_results_data if not r['is_correct']]
if failed_cases:
    print("\n" + "=" * 70)
    print("[ANALYSIS] Failed Cases Detail:")
    for i, result in enumerate(failed_cases[:10], 1):  # Show top 10 failures
        print(f"\nCase {i}:")
        print(f"  Query: {result['query']}")
        print(f"  Type: {result['query_type']}")
        print(f"  Expected: {result['expected_name']} (Rank #{result['rank']})")
        print(f"  Predicted: {result['top1_name']} (Score: {result['top1_score']:.4f})")
        print(f"  Top3: {' > '.join([f'{name}({score:.3f})' for _, score, name in result['top3']])}")

# ========== FINAL CONCLUSIONS ==========
print("\n" + "=" * 70)
print("[CONCLUSION] Evaluation Summary:")
print(f"Model: {MODEL_NAME} ({embedding_dim}D)")
top1_acc = calculate_top_k_accuracy(all_results_data, 1)

if top1_acc >= 0.85:
    print("\n[RESULT] EXCELLENT - Model meets requirements for personalized retrieval")
    print("  - Effectively distinguishes between functionally similar items")
    print("  - Handles personalization and context well")
elif top1_acc >= 0.70:
    print("\n[RESULT] ACCEPTABLE - Good baseline, but refinement needed")
    print("  - May struggle with very subtle functional distinctions")
else:
    print("\n[RESULT] INSUFFICIENT - Needs improvement")
    print("  - Consider better attribute engineering or larger models")

# ========== SAVE RESULTS ==========
output_dir = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(output_dir, "final_evaluation_results_english.csv")

df = pd.DataFrame([{
    'Query': r['query'],
    'Query_Type': r['query_type'],
    'Expected': r['expected_name'],
    'Predicted': r['top1_name'],
    'Score': r['top1_score'],
    'Rank': r['rank'],
    'Correct': 'Yes' if r['is_correct'] else 'No',
    'Latency_ms': r['latency_ms'],
    'Confidence_Gap': r['top1_score'] - r['top2_score']
} for r in all_results_data])

df.to_csv(results_file, index=False, encoding='utf-8-sig')
print(f"\n[OUTPUT] Detailed results saved to: {results_file}")
