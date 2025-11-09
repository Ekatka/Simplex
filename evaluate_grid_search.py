"""
Script to evaluate all models from grid search and find the best one.
Compares models based on:
- Number of steps to solve (fewer is better)
- Game value accuracy
- Training time
"""

import os
import json
import glob
from stable_baselines3 import PPO
from matrix import Matrix
from envs import RandomMatrixEnv
from config import M, N, MIN_VAL, MAX_VAL, EPSILON, PIVOT_MAP
from base_matrix import BASE_MATRIX
from testing import (
    extract_optimal_strategy,
    extract_second_player_strategy,
    compute_game_value_from_strategies
)


def evaluate_model(model_path: str, matrix: Matrix, num_runs: int = 5):
    """
    Evaluate a single model by running it multiple times and averaging results.
    
    Returns:
        dict with average steps, game values, and success rate
    """
    try:
        model = PPO.load(model_path)
    except Exception as e:
        return {"error": str(e)}
    
    steps_list = []
    game_values = []
    successful_runs = 0
    
    for _ in range(num_runs):
        env = RandomMatrixEnv(matrix)
        obs, _ = env.reset()
        done = False
        max_steps = 2000  # Safety limit
        
        while not done and env.nit < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        
        if done:
            successful_runs += 1
            steps_list.append(env.nit)
            
            # Extract strategies and compute game value
            first_player_strategy = extract_optimal_strategy(env.T, env.basis, M)
            second_player_strategy = extract_second_player_strategy(env.T, env.basis, M, N)
            game_value = compute_game_value_from_strategies(matrix, first_player_strategy, second_player_strategy)
            game_values.append(game_value)
    
    if successful_runs == 0:
        return {"error": "Model failed to solve in all runs"}
    
    return {
        "avg_steps": sum(steps_list) / len(steps_list),
        "min_steps": min(steps_list),
        "max_steps": max(steps_list),
        "avg_game_value": sum(game_values) / len(game_values),
        "success_rate": successful_runs / num_runs,
        "num_successful": successful_runs
    }


def find_best_model_from_json(json_path: str, num_test_runs: int = 5):
    """
    Load grid search results from JSON and evaluate all models.
    """
    print(f"Loading grid search results from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    print(f"Found {len(results)} model configurations to evaluate\n")
    
    # Initialize matrix (same as training)
    matrix = Matrix(m=M, n=N, min=MIN_VAL, max=MAX_VAL, epsilon=EPSILON, base_P=BASE_MATRIX)
    
    # Evaluate each model
    evaluations = []
    for i, result in enumerate(results, 1):
        if "error" in result:
            print(f"[{i}/{len(results)}] Skipping failed run: {result.get('error', 'Unknown error')}")
            continue
        
        model_path = result.get("model_path")
        if not model_path or not os.path.exists(model_path):
            print(f"[{i}/{len(results)}] Model file not found: {model_path}")
            continue
        
        print(f"[{i}/{len(results)}] Evaluating: lr={result['learning_rate']:.2e}, "
              f"n_steps={result['n_steps']}, clip_range={result['clip_range']}")
        
        eval_result = evaluate_model(model_path, matrix, num_test_runs)
        
        if "error" in eval_result:
            print(f"  ERROR: {eval_result['error']}\n")
            evaluations.append({
                **result,
                "evaluation_error": eval_result["error"]
            })
        else:
            print(f"  Avg steps: {eval_result['avg_steps']:.1f} "
                  f"(min: {eval_result['min_steps']}, max: {eval_result['max_steps']})")
            print(f"  Success rate: {eval_result['success_rate']*100:.1f}% "
                  f"({eval_result['num_successful']}/{num_test_runs})")
            print(f"  Avg game value: {eval_result['avg_game_value']:.6f}\n")
            
            evaluations.append({
                **result,
                **eval_result
            })
    
    return evaluations


def print_best_models(evaluations):
    """
    Print summary of best models sorted by different criteria.
    """
    # Filter out failed evaluations
    valid_evals = [e for e in evaluations if "evaluation_error" not in e and "error" not in e]
    
    if not valid_evals:
        print("No successful evaluations found!")
        return
    
    print("\n" + "="*100)
    print("BEST MODELS SUMMARY")
    print("="*100)
    
    # Sort by average steps (fewer is better)
    sorted_by_steps = sorted(valid_evals, key=lambda x: x.get("avg_steps", float('inf')))
    
    print("\n🏆 TOP 5 MODELS BY AVERAGE STEPS (fewer is better):")
    print("-" * 100)
    print(f"{'Rank':<6} {'LR':<12} {'N_STEPS':<10} {'CLIP':<8} {'Avg Steps':<12} {'Success Rate':<15} {'Model Path':<50}")
    print("-" * 100)
    for i, model in enumerate(sorted_by_steps[:5], 1):
        print(f"{i:<6} {model['learning_rate']:<12.2e} {model['n_steps']:<10} "
              f"{model['clip_range']:<8.1f} {model.get('avg_steps', 'N/A'):<12.1f} "
              f"{model.get('success_rate', 0)*100:<14.1f}% "
              f"{os.path.basename(model['model_path']):<50}")
    
    # Sort by success rate
    sorted_by_success = sorted(valid_evals, key=lambda x: x.get("success_rate", 0), reverse=True)
    
    print("\n✅ TOP 5 MODELS BY SUCCESS RATE:")
    print("-" * 100)
    print(f"{'Rank':<6} {'LR':<12} {'N_STEPS':<10} {'CLIP':<8} {'Success Rate':<15} {'Avg Steps':<12} {'Model Path':<50}")
    print("-" * 100)
    for i, model in enumerate(sorted_by_success[:5], 1):
        print(f"{i:<6} {model['learning_rate']:<12.2e} {model['n_steps']:<10} "
              f"{model['clip_range']:<8.1f} {model.get('success_rate', 0)*100:<14.1f}% "
              f"{model.get('avg_steps', 'N/A'):<12.1f} "
              f"{os.path.basename(model['model_path']):<50}")
    
    # Overall best (combination of steps and success rate)
    def score_model(m):
        steps = m.get("avg_steps", float('inf'))
        success = m.get("success_rate", 0)
        # Lower steps is better, higher success is better
        # Score = success_rate / (steps + 1) to avoid division by zero
        return success / (steps + 1)
    
    sorted_by_score = sorted(valid_evals, key=score_model, reverse=True)
    
    print("\n⭐ OVERALL BEST MODELS (balanced score: success_rate / steps):")
    print("-" * 100)
    print(f"{'Rank':<6} {'LR':<12} {'N_STEPS':<10} {'CLIP':<8} {'Score':<12} {'Avg Steps':<12} {'Success Rate':<15} {'Model Path':<50}")
    print("-" * 100)
    for i, model in enumerate(sorted_by_score[:5], 1):
        score = score_model(model)
        print(f"{i:<6} {model['learning_rate']:<12.2e} {model['n_steps']:<10} "
              f"{model['clip_range']:<8.1f} {score:<12.4f} "
              f"{model.get('avg_steps', 'N/A'):<12.1f} "
              f"{model.get('success_rate', 0)*100:<14.1f}% "
              f"{os.path.basename(model['model_path']):<50}")
    
    print("\n" + "="*100)
    print(f"BEST MODEL: {os.path.basename(sorted_by_score[0]['model_path'])}")
    print(f"  Learning rate: {sorted_by_score[0]['learning_rate']:.2e}")
    print(f"  N steps: {sorted_by_score[0]['n_steps']}")
    print(f"  Clip range: {sorted_by_score[0]['clip_range']}")
    print(f"  Average steps: {sorted_by_score[0].get('avg_steps', 'N/A'):.1f}")
    print(f"  Success rate: {sorted_by_score[0].get('success_rate', 0)*100:.1f}%")
    print("="*100)


def main():
    import sys
    
    # Find the most recent grid search results file
    json_files = glob.glob("models/grid_search_results_*.json")
    
    if not json_files:
        print("No grid search results found!")
        print("Please run grid search first: python train.py --grid-search")
        return
    
    # Use the most recent file
    json_path = max(json_files, key=os.path.getctime)
    
    num_runs = 5
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of runs: {sys.argv[1]}, using default: 5")
    
    print(f"Evaluating models with {num_runs} test runs each...")
    print(f"Using results file: {json_path}\n")
    
    evaluations = find_best_model_from_json(json_path, num_runs)
    
    # Save evaluation results
    eval_output = json_path.replace("grid_search_results", "evaluation_results")
    with open(eval_output, 'w') as f:
        json.dump(evaluations, f, indent=2)
    print(f"\nEvaluation results saved to: {eval_output}")
    
    print_best_models(evaluations)


if __name__ == "__main__":
    main()

