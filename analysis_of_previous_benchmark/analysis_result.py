import datasets
import os
import argparse
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for mean calculation

def parse_args():
    parser=argparse.ArgumentParser(description="Summarize and visualize model evaluation results.")
    parser.add_argument('--result_dataset_path', required=True, help="Path to the parent directory containing evaluation results.")
    parser.add_argument('--output_dir', default='summary_plots', help="Directory to save the generated plots and summary CSV.")
    return parser.parse_args()

if __name__=='__main__':
    
    model_names = [ # Renamed to model_names for clarity
        'Mistral-Large-Instruct-2411',
        'Qwen2.5-72B-Instruct',
        'Gemma-3-27b-it',
        'Llama-4-Scout-17B-16E-Instruct',
        'Athene-V2-Chat',
        'GLM-4-32B-0414',
        'Command-a',
    ]
    avaliable_dataset = ['doc2dial','inscit','qrecc','quac','topiocqa','coral',]
    # Add more score types here if needed
    score_names = ['answerable_score','tag_score','atomic_score','reference_score','faithful_score','quality_score'] 

    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = [] # Store results for all combinations

    logger.info("Starting result aggregation...")
    for judge_model_name in model_names:
        for tested_model_name in avaliable_dataset:
            for score_name in score_names:
                dataset_dir = os.path.join(args.result_dataset_path, f'{judge_model_name}_{tested_model_name}')
                try:
                    # Check if directory exists before trying to load
                    if not os.path.exists(dataset_dir):
                        logger.warning(f"Dataset directory not found, skipping: {dataset_dir}")
                        all_results.append({
                            'judge_model': judge_model_name,
                            'tested_dataset': tested_model_name,
                            'score_name': score_name,
                            'average_score': np.nan, # Use NaN for missing data
                            'num_scores': 0
                        })
                        continue # Skip to the next iteration
                        
                    dataset = datasets.load_from_disk(dataset_dir)
                    
                    # Check if the score column exists
                    if score_name not in dataset.column_names:
                        logger.warning(f"Score column '{score_name}' not found in {dataset_dir}, skipping.")
                        all_results.append({
                            'judge_model': judge_model_name,
                            'tested_model': tested_model_name,
                            'score_name': score_name,
                            'average_score': np.nan, 
                            'num_scores': 0
                        })
                        continue

                    scores = dataset[score_name] # This should be a list of integers
                    print(judge_model_name,tested_model_name,score_name,scores.count(None))
                    # Ensure scores is a list and contains numbers
                    if not isinstance(scores, list):
                         logger.warning(f"Column '{score_name}' in {dataset_dir} is not a list, skipping.")
                         # Handle potential case where column exists but isn't a list as expected
                         average_score = np.nan
                         num_scores = 0
                    elif not scores: # Handle empty list
                        logger.warning(f"Score list '{score_name}' is empty in {dataset_dir}.")
                        average_score = np.nan # Or maybe 0, depending on desired handling
                        num_scores = 0
                    else:
                        # Calculate the average score
                        # Filter out potential non-numeric types if necessary, though ideally it's clean
                        numeric_scores = [s for s in scores if isinstance(s, (int, float))]
                        if not numeric_scores:
                             logger.warning(f"No numeric scores found in '{score_name}' list in {dataset_dir}.")
                             average_score = np.nan
                             num_scores = 0
                        else:
                            average_score = np.mean(numeric_scores)
                            num_scores = len(numeric_scores)
                            if len(numeric_scores) != len(scores):
                                logger.warning(f"Non-numeric values found and ignored in '{score_name}' list in {dataset_dir}.")

                    # Store the aggregated result
                    all_results.append({
                        'judge_model': judge_model_name,
                        'tested_model': tested_model_name,
                        'score_name': score_name,
                        'average_score': average_score,
                        'num_scores': num_scores
                    })
                    # logger.info(f"Processed: {judge_model_name} vs {tested_model_name} for {score_name} -> Avg: {average_score:.2f} ({num_scores} scores)")

                except Exception as e:
                    logger.error(f"Failed to process {dataset_dir} for score {score_name}: {e}")
                    # Append with NaN even if there's an unexpected error during loading/processing
                    all_results.append({
                        'judge_model': judge_model_name,
                        'tested_model': tested_model_name,
                        'score_name': score_name,
                        'average_score': np.nan,
                        'num_scores': 0
                    })

    logger.info("Aggregation complete. Converting to DataFrame...")
    if not all_results:
        logger.error("No results were aggregated. Exiting.")
        exit()
        
    # Convert the list of results into a Pandas DataFrame
    results_df = pd.DataFrame(all_results)

    # Save the raw aggregated data
    summary_csv_path = os.path.join(args.output_dir, 'aggregated_scores_summary.csv')
    results_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Aggregated summary saved to {summary_csv_path}")

    # --- Visualization ---
    logger.info("Generating visualizations...")

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Generate plots for each score type
    for score_name in results_df['score_name'].unique():
        logger.info(f"--- Generating plots for score: {score_name} ---")
        
        # Filter data for the current score type
        score_df = results_df[results_df['score_name'] == score_name]

        # 1. Heatmap of Average Scores (Judge vs Tested)
        try:
            heatmap_data = score_df.pivot(index='judge_model', columns='tested_model', values='average_score')
            
            # Ensure the order of rows/columns matches model_names for consistency
            heatmap_data = heatmap_data.reindex(index=model_names, columns=avaliable_dataset)

            plt.figure(figsize=(12, 10))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, cbar_kws={'label': f'Average {score_name}'})
            plt.title(f'Heatmap of Average {score_name}\n(Judge Model vs Tested Dataset)')
            plt.xlabel('Tested Dataset')
            plt.ylabel('Judge Model')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            heatmap_filename = os.path.join(args.output_dir, f'{score_name}_heatmap.png')
            plt.savefig(heatmap_filename)
            logger.info(f"Saved heatmap to {heatmap_filename}")
            # plt.show() # Uncomment if you want to display plots interactively
            plt.close() # Close the plot to free memory

        except Exception as e:
            logger.error(f"Failed to generate heatmap for {score_name}: {e}")


        # 2. Bar Chart: Average Score Received by each Tested Dataset
        try:
            avg_score_received = score_df.groupby('tested_model')['average_score'].mean().reindex(avaliable_dataset).sort_values(ascending=False)
            plt.figure(figsize=(12, 7)) # Keep a slightly larger figure for readability
            ax = avg_score_received.plot(kind='bar', color=sns.color_palette("viridis", len(avaliable_dataset)))
            plt.title(f'Average {score_name} Received per Model (Averaged across all Judges)')
            plt.xlabel('Tested Dataset')
            plt.ylabel(f'Average {score_name}')
            plt.xticks(rotation=45, ha='right')

            for bar in ax.patches:
                height = bar.get_height()
                if pd.isna(height):
                    continue
                ax.text(
                    x=bar.get_x() + bar.get_width() / 2.,
                    y=height,
                    s=f'{height:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            
            # Use original y-axis limits logic
            min_val_received = avg_score_received.min(skipna=True) if not avg_score_received.empty else 0
            max_val_received = avg_score_received.max(skipna=True) if not avg_score_received.empty else 1
            plt.ylim(bottom=max(0, min_val_received - 0.1), 
                     top=(max_val_received + 0.1 if not pd.isna(max_val_received) else 1)) # Ensure top limit is sensible even if max_val is NaN

            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            received_filename = os.path.join(args.output_dir, f'{score_name}_avg_score_received.png')
            plt.savefig(received_filename)
            logger.info(f"Saved average score received plot to {received_filename}")
            plt.close()

        except Exception as e:
            logger.error(f"Failed to generate average score received plot for {score_name}: {e}")


        # 3. Bar Chart: Average Score Given by each Judge Model
        try:
            # Calculate the mean score given by each judge, ignoring NaNs
            avg_score_given = score_df.groupby('judge_model')['average_score'].mean().reindex(avaliable_dataset).sort_values(ascending=False)
            plt.figure(figsize=(12, 7)) # Keep a slightly larger figure for readability
            ax_given = avg_score_given.plot(kind='bar', color=sns.color_palette("magma", len(avaliable_dataset)))
            plt.title(f'Average {score_name} Given per Judge Model (Averaged across all Tested Datasets)')
            plt.xlabel('Judge Model')
            plt.ylabel(f'Average {score_name}')
            plt.xticks(rotation=45, ha='right')

            for bar in ax_given.patches:
                height = bar.get_height()
                if pd.isna(height):
                    continue
                ax_given.text(
                    x=bar.get_x() + bar.get_width() / 2.,
                    y=height,
                    s=f'{height:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            # Use original y-axis limits logic
            min_val_given = avg_score_given.min(skipna=True) if not avg_score_given.empty else 0
            max_val_given = avg_score_given.max(skipna=True) if not avg_score_given.empty else 1
            plt.ylim(bottom=max(0, min_val_given - 0.1), 
                     top=(max_val_given + 0.1 if not pd.isna(max_val_given) else 1)) # Ensure top limit is sensible

            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            given_filename = os.path.join(args.output_dir, f'{score_name}_avg_score_given.png')
            plt.savefig(given_filename)
            logger.info(f"Saved average score given plot to {given_filename}")
            plt.close()
            
        except Exception as e:
             logger.error(f"Failed to generate average score given plot for {score_name}: {e}")

    logger.info("Visualization generation complete.")