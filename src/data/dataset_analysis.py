import pandas as pd
import numpy as np
import os, os.path
from glob import glob
import config as cfg

def build_bird_dict(bird_list):
    bird_dict = {}
    for bird_count, bird_name in enumerate(sorted(bird_list)):
        bird_dict[bird_name] = bird_count
    return bird_dict

def build_bird_time_dict(bird_list):
    bird_dict = {}
    for bird_count, bird_name in enumerate(sorted(bird_list)):
        bird_dict[bird_name] = []
    return bird_dict

def add_occurence(occurence_matrix, current_bird_list, bird_dict):
    for target_bird_index, target_bird in enumerate(current_bird_list):
        target_bird_id = bird_dict[target_bird]
        for other_bird_index, other_bird in enumerate(current_bird_list):
            other_bird_id = bird_dict[other_bird]
            if(target_bird_index != other_bird_index):
                # pass
                occurence_matrix[target_bird_id][other_bird_id] += 1

if __name__ == "__main__":
    annotation_dir = cfg.annotation_dir
    annotation_list = glob(os.path.join(annotation_dir, "*.txt"))
    bird_list = cfg.bird_list
    bird_dict = build_bird_dict(bird_list)
    
    # =========================================================================================
    # occurence ratio matrix calaulated
    # =========================================================================================
    occurence_matrix = np.zeros((len(bird_dict), len(bird_dict)))
    for file_count, annotation_file in enumerate(annotation_list):
        current_df = pd.read_csv(annotation_file, sep="\t")
        current_bird_list =  list(current_df["Species"])
        add_occurence(occurence_matrix, current_bird_list, bird_dict)
    occurence_matrix = occurence_matrix/occurence_matrix.sum(axis=1)[:,None]
    occurence_df = pd.DataFrame(occurence_matrix, columns=sorted(bird_list), index=sorted(bird_list))
    occurence_df.to_csv("../occurence_analysis.csv", float_format='%.3f')
    # print(np.matrix(occurence_matrix))
    # for row in occurence_matrix:
    #     for val in row:
    #         print ('{:4}'.format(val), end=" ")
    #     print()
    # print(species_list)

    # =========================================================================================
    # average time of each species
    # =========================================================================================
    bird_time_dict = build_bird_time_dict(bird_list)
    for file_count, annotation_file in enumerate(annotation_list):
        current_df = pd.read_csv(annotation_file, sep="\t")
        current_bird_set =  set(list(current_df["Species"]))
        for bird in current_bird_set:
            extract_df = current_df[current_df["Species"] == bird]
            # print(extract_df)
            for index, row in extract_df.iterrows():
                duration = row["End Time (s)"] - row["Begin Time (s)"]
                bird_time_dict[bird].append(duration)

    time_analysis_df = pd.DataFrame(columns=["Name", "Average", "Deviation", "Max", "Min"])
    for bird in sorted(bird_list):
        current_list = bird_time_dict[bird]
        new_row = {"Name":bird, "Average":np.average(current_list), "Deviation":np.std(current_list), "Max":np.max(current_list), "Min":np.min(current_list)}
        time_analysis_df = time_analysis_df.append(new_row, ignore_index=True)
    time_analysis_df.to_csv("../dataset_time_analysis.csv", index=False, float_format='%.3f')
    # print(bird_time_dict["RCKI"])
