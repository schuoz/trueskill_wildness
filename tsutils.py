# import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import json
import random
from trueskill import Rating, quality_1vs1, rate_1vs1, TrueSkill


def process_matches(players):
    player_ids = pd.concat([players.geo_B_id,players.geo_A_id]).unique()

    # Create a DataFrame to store the related table
    related_table = pd.DataFrame(columns=['player_id', 'player_ind'])

    # Populate the related table with player IDs and their corresponding indices
    for i, player_id in enumerate(player_ids):
        related_table = related_table.append({'player_id': player_id,
                                              'player_ind': i}, ignore_index=True)

    num_players = len(player_ids)

    # Initialize player ratings
    player_ratings = [Rating() for i in range(num_players)]

    # Iterate through each match in the round
    for _, match in players.iterrows():
        winner_id = match['winner_ID']
        loser_id = match['loser_ID']

        # Find the winner and loser indices from the related table
        winner_row = related_table[related_table['player_id'] == winner_id]
        loser_row = related_table[related_table['player_id'] == loser_id]

        if not winner_row.empty and not loser_row.empty:
            winner_id_ind = winner_row.iloc[0]['player_ind']
            loser_id_ind = loser_row.iloc[0]['player_ind']
        else:
            print("Winner ID or Loser ID not found in the related table.")
        
        # Simulate the match and update player ratings
        player_ratings[winner_id_ind], player_ratings[loser_id_ind] = rate_1vs1(player_ratings[winner_id_ind], player_ratings[loser_id_ind])

    # Add ratings to the related table
    for i, rating in enumerate(player_ratings):
        related_table.loc[related_table['player_ind'] == i, 'wildness_mean'] = rating.mu

    return related_table


def process_wildness(task_merge, related_table):
    A_task = task_merge[["geo_A_id","geo_A_lat","geo_A_lon"]]
    A_task.rename(columns={'geo_A_id': 'player_id'}, inplace=True)

    A_wildness = pd.merge(related_table, A_task, on = ["player_id"])
    A_wildness.drop_duplicates(inplace=True)
    A_wildness.rename(columns={'geo_A_lat': 'lat', 'geo_A_lon': 'lon'}, inplace=True)

    B_task = task_merge[["geo_B_id","geo_B_lat","geo_B_lon"]]
    B_task.rename(columns={'geo_B_id': 'player_id'}, inplace=True)

    B_wildness = pd.merge(related_table, B_task, on = ["player_id"])
    B_wildness.drop_duplicates(inplace=True)
    B_wildness.rename(columns={'geo_B_lat': 'lat', 'geo_B_lon': 'lon'}, inplace=True)

    wildness = pd.concat([A_wildness,B_wildness])
    wildness.drop_duplicates(inplace=True)

    return wildness

def reorder_dataframe(df, id_column):
    """
    Reorders a DataFrame based on a specified ID column,
    ensuring each unique ID value is represented evenly
    throughout the reordered DataFrame.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be reordered.
    - id_column: str
        The name of the ID column based on which the DataFrame
        should be reordered.

    Returns:
    - reordered_df: pandas DataFrame
        The reordered DataFrame.
    """
    # Sort the DataFrame based on the specified ID column
    df_sorted = df.sort_values(by=id_column)

    # Determine the number of unique IDs
    num_unique_ids = df_sorted[id_column].nunique()

    # Calculate the maximum number of times any unique ID appears in the dataset
    max_repeat_count = df_sorted[id_column].value_counts().max()

    # Create a list to hold the reordered data
    reordered_data = []

    # Reorder the DataFrame by cycling through the data
    for i in range(max_repeat_count):
        for id_value in df_sorted[id_column].unique():
            id_subset = df_sorted[df_sorted[id_column] == id_value]
            if len(id_subset) > i:
                id_subset['iteration'] = i + 1
                reordered_data.append(id_subset.iloc[i])

    # Concatenate the reordered subsets into a single DataFrame
    reordered_df = pd.DataFrame(reordered_data)

    # Reset the index of the reordered DataFrame
    reordered_df.reset_index(drop=True, inplace=True)

    return reordered_df

# compute convergence

class MatchSimulator:
    def __init__(self, matches, player_ids):
        self.matches = matches
        self.player_ids = player_ids
        self.num_players = len(player_ids)
        self.player_ratings = [Rating() for _ in range(self.num_players)]
        self.convergence_data = [[] for _ in range(len(self.player_ratings))]
        self.prev_ratings = [rating.mu for rating in self.player_ratings]
        self.iteration_convergence_data = []

    def update_ratings(self, winner_id, loser_id):
        winner_id_ind = int(np.where(self.player_ids == winner_id)[0])
        loser_id_ind = int(np.where(self.player_ids == loser_id)[0])
        self.player_ratings[winner_id_ind], self.player_ratings[loser_id_ind] = rate_1vs1(self.player_ratings[winner_id_ind], self.player_ratings[loser_id_ind])

    def check_convergence(self):
        converged_players = [abs(rating.mu - prev_rating) < 0.1 for rating, prev_rating in zip(self.player_ratings, self.prev_ratings)]
        convergence_percentage = sum(converged_players) / len(converged_players)
        return convergence_percentage

    def simulate(self):
        unique_rounds = self.matches['iteration'].unique()

        for iterations, round_num in enumerate(unique_rounds):
            round_matches = self.matches[self.matches['iteration'] == round_num]

            for _, match in round_matches.iterrows():
                self.update_ratings(match['winner_ID'], match['loser_ID'])

            for i, rating in enumerate(self.player_ratings):
                self.convergence_data[i].append(rating.mu)

            self.iteration_convergence_data.append([abs(rating.mu - prev_rating) for rating, prev_rating in zip(self.player_ratings, self.prev_ratings)])

            convergence_percentage = self.check_convergence()
            print(f"Iteration {iterations}: Convergence Percentage = {convergence_percentage}")

            self.prev_ratings = [rating.mu for rating in self.player_ratings]

            if iterations >= 31:
                print(f"Iteration {iterations}: All rounds completed.")
                break

        return self.player_ratings, self.convergence_data, self.iteration_convergence_data