import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import normalize
from transformers import AutoTokenizer

from multiprocessing.pool import ThreadPool
import numpy as np
from tqdm import tqdm
import os
import json
import re
from unidecode import unidecode
from cleantext import clean

import reverse_geocoder
from pycountry_convert import country_alpha2_to_continent_code

country_to_code_map = {
    "Belgium"       : "BE", 	
    "Greece"        : "EL", 	
    "ithuania"      : "LT", 	
    "Portugal"      : "PT",
    "Bulgaria"      : "BG", 
    "Spain"         : "ES", 	
    "Luxembourg"    : "LU", 	
    "Romania"       : "RO",
    "Czechia"       : "CZ", 	
    "France"        : "FR", 	
    "Hungary"       : "HU", 	
    "Slovenia"      : "SI",
    "Denmark"       : "DK", 	
    "Croatia"       : "HR", 	
    "Malta"         : "MT", 	
    "Slovakia"      : "SK",
    "Germany"       : "DE", 	
    "Italy"         : "IT", 	
    "Netherlands"   : "NL", 	
    "Finland"       : "FI",
    "Estonia"       : "EE", 	
    "Cyprus"        : "CY", 	
    "Austria"       : "AT", 
    "Sweden"        : "SE",
    "Ireland"       : "IE", 	
    "Latvia"        : "LV", 	
    "Poland"        : "PL",
    "Canada"        : "CA",
    "United States" : "US"
}

code_to_country_map = {j:i for i,j in country_to_code_map.items()}

continent_code_to_name_map = {
    "AS": "Asia",
    "NA": "North America",
    "SA": "South America",
    "EU": "Europe",
    "AF": "Africa",
    "OC": "Oceania"  
}

continent_code_to_code_map = {
    i:j for i, j in zip(continent_code_to_name_map.keys(), list(range(6)))
}


def collate_fn(tensor):
    x_batch = [i[0] for i in tensor]
    y_batch = torch.vstack([i[1] for i in tensor]).cuda()
    x_batch = pad_sequence(x_batch).cuda().squeeze()

    # Shorten long sequences
    if x_batch.shape[0] > 100:
        x_batch = x_batch[-100:, :, :]

    return (x_batch, y_batch)


def obs_to_tensor(obs):
    """One hot encodes a string to a Tensor"""
    n_letters = len(characters_to_keep)
    tensor = torch.zeros(len(obs), 1, n_letters)
    for idx, char in enumerate(obs):
        tensor[idx][0][characters_to_keep.find(char)] = 1
    return tensor


def find_characters_to_keep(data, max_n_characters=100):
    """Finds the most frequent charachters in an iterable"""
    character_count_map = {}
    for obs in data:
        for char in obs:
            if char not in character_count_map.keys():
                character_count_map[char] = 0
            character_count_map[char] += 1

    character_count_sorted = sorted(character_count_map.items(),
                                 key=lambda x: x[1],
                                 reverse=True)

    global characters_to_keep
    characters_to_keep = "abcdefghijklmnopqrstuvwxyz"
    characters_to_keep += "12334567890"

    # Add the most frequent characters to characters_to_keep
    while (len(characters_to_keep) <= max_n_characters
          and len(character_count_sorted) > 0):
        new_char = character_count_sorted.pop(0)[0]
        if new_char not in characters_to_keep:
            characters_to_keep += new_char

    # Remove these characters regardless
    characters_to_remove = """@.,#'_-":);(/&*`$%|[]+~><=^\{}"""
    for i in characters_to_remove:
        characters_to_keep = characters_to_keep.replace(i, "")

    print("Characters to keep: ", characters_to_keep)

    return characters_to_keep


def undo_min_max_scaling(tensor):
    # Read information about the scaling previously applied
    with open("data/temp/scaling_info.txt") as file:
        data = file.readlines()
        data = [float(i.replace("\n", "")) for i in data]
        min_y1 = data[0]
        max_y1 = data[1]
        min_y2 = data[2]
        max_y2 = data[3]

    y1 = tensor[:, 0]
    y2 = tensor[:, 1]

    # Undo the min max scaling
    y1_unscaled = y1 * (max_y1 - min_y1) + min_y1
    y2_unscaled = y2 * (max_y2 - min_y2) + min_y2

    return np.vstack([y1_unscaled, y2_unscaled]).transpose()


class LocationDataset(Dataset):

    def __init__(self, max_obs):
        self.x_data = []
        self.y_data = []
                                          
        self.load_data()
        self.reduce(max_obs=max_obs)
        self.add_country_continent_code()
        self.min_max_scale_y_data()
        self.remove_special_characters()
        self.lower()
        self.remove_short()
        self.write()
                                          
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.x_data = self.tokenizer(self.x_data.tolist(), padding=True)

    def load_data(self):
        train_data_path = "data/training_data/training_data/"
        for file in tqdm(os.listdir(train_data_path), desc="Reading data"):
            with open(train_data_path + file, "r") as file:
                file_data = file.readlines()
                for obs in file_data:
                    obs_dict = json.loads(obs)
                    self.x_data.append(obs_dict["text"])
                    self.y_data.append(obs_dict["coordinates"])
    
        self.x_data = np.array(self.x_data)
        self.y_data_unnorm = [(float(i[0]), float(i[1])) for i in self.y_data]
        self.y_data_unnorm = torch.Tensor(self.y_data_unnorm)

    def reduce(self, max_obs):
        # Reduce
        if max_obs != None: 
            rand_idx = np.random.randint(low=0, high=len(self.x_data),
                                         size=max_obs)
            self.x_data = self.x_data[rand_idx]
            self.y_data_unnorm = self.y_data_unnorm[rand_idx]

    def add_country_continent_code(self):
        coordinates = self.y_data_unnorm.tolist()
        coordinates = [(i[1], i[0]) for i in coordinates]
        countries = reverse_geocoder.search(coordinates)
        self.country = np.array([i["cc"] for i in countries])
        self.country = np.char.replace(self.country, "TL", "ID")
        self.country = np.char.replace(self.country, "EH", "MA")
        self.continent = np.array([country_alpha2_to_continent_code(i)
                                  for i in self.country])
        self.continent_name = np.array([continent_code_to_name_map[i]
                                       for i in self.continent])
        self.continent = np.array([continent_code_to_code_map[i]
                                   for i in self.continent])
        self.continent = torch.vstack([torch.eye(n=6)[:, i] for i in self.continent])

    def min_max_scale_y_data(self):
        """Applies min max scaling to the coordinates"""
        self.max_y1 = self.y_data_unnorm[:, 0].max()
        self.min_y1 = self.y_data_unnorm[:, 0].min()
        self.max_y2 = self.y_data_unnorm[:, 1].max()
        self.min_y2 = self.y_data_unnorm[:, 1].min()

        y1_scaled = (self.y_data_unnorm[:, 0] - self.min_y1) / (self.max_y1 - self.min_y1)
        y2_scaled = (self.y_data_unnorm[:, 1] - self.min_y2) / (self.max_y2 - self.min_y2)

        self.y_data = torch.vstack([y1_scaled, y2_scaled]).t()

    def remove_special_characters(self):
        # Replace line breaks and tabs
        self.x_data = np.char.replace(self.x_data, "\n", "")
        self.x_data = np.char.replace(self.x_data, "\t", "")
        # Replace links
        self.x_data = np.array([re.sub(r'http\S+', '', i) for i in self.x_data])
        # Replace tags
        self.x_data = np.array([re.sub(r'@\S+', '', i) for i in self.x_data])
        self.x_data = np.array([re.sub(r'#\S+', '', i) for i in self.x_data])
        # Also replace the following special characters
        special_characters = [
            "@", "(", ")", ":", "/", ".", "&", "=", "-", "'", '"', "/", "\\",
            "!", "?", "`"
        ]
        to_replace = {i:"" for i in special_characters}
        self.x_data = np.array([i.translate(i.maketrans(to_replace)) for i in self.x_data])
        # Remove emoji
        self.x_data = np.array([clean(i, no_emoji=True) for i in self.x_data])

    def unidecode(self):
        # Unicode decode
        self.x_data = np.array([unidecode(i) for i in
                                tqdm(self.x_data, desc="Unidecode")])

    def lower(self):
        # All letters to lower case
        self.x_data = np.char.lower(self.x_data)

    def sort(self):
        # Order from short to long sentences for efficient batching
        obs_len = np.char.str_len(self.x_data)
        str_len_order = obs_len.argsort()
        self.x_data = self.x_data[str_len_order]
        self.y_data = self.y_data[str_len_order]
        self.country = self.country[str_len_order]

    def remove_short(self):
        # Remove empty and short observations
        obs_len = np.char.str_len(self.x_data)
        self.x_data = self.x_data[np.where(obs_len > 9)[0]]
        self.y_data = self.y_data[np.where(obs_len > 9)[0]]
        self.country = self.country[np.where(obs_len > 9)[0]]

    def add_weights(self):
        # Add weights based on country
        # class_weights_map = {} 
        # for country in np.unique(self.country):
        #     class_weights_map[country] = \
        #         len(self.country[self.country == country]) / len(self.country)

        # class_weights_map = {k:v/max(class_weights_map.values()) for k,v
        #                      in class_weights_map.items()}
        # self.weights = [class_weights_map[i] for i in self.country]
        pass

    def write(self):
        """Writes self.x_data, self.y_data and self.country to the disk"""
        np.savetxt("data/temp/x_data.txt", self.x_data, fmt="%s")
        np.savetxt("data/temp/y_data.txt", self.y_data, fmt="%s")
        np.savetxt("data/temp/country.txt", self.country, fmt="%s")
        np.savetxt("data/temp/continent.txt", self.continent, fmt="%s")
        np.savetxt("data/temp/scaling_info.txt",
                   np.array([self.min_y1, self.max_y1, self.min_y2, self.max_y2]),
                   fmt="%s")

    def to_tensor(self):
        """Transforms the strings x data into one hot encoded tensor on the
        character level"""
        with ThreadPool(10) as pool:
            x_data_encoded = pool.map(obs_to_tensor, self.x_data)

        self.x_data_raw = self.x_data
        self.x_data = x_data_encoded

    def find_most_popular_words(self):
        # Find the most popular words
        words = "".join([i.strip() for i in self.x_data]).split()
        words, counts = np.unique(words, return_counts=True)

        words = words[counts.argsort()[::-1]]
        counts = counts[counts.argsort()[::-1]]

        # Write them to the disk
        with open("data/temp/most_popular_words.txt", "w") as file:
            file.writelines([i + " " + str(j) + "\n" for i, j in zip(words, counts)])

    def __len__(self):
        return len(self.continent)

    def __getitem__(self, index):
        input_ids = torch.Tensor(self.x_data[index].ids).to(int)
        attention_mask = torch.Tensor(self.x_data[index].attention_mask).to(int)
        y = self.continent[index, :].to(int)

        return (input_ids, attention_mask, y)