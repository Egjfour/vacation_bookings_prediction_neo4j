"""
build_data will be used to create the dataset for this experiment
We will be able to form relationships and ensure data integrity in this way
"""
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
np.random.seed(123)

class DataBuilder(ABC):
    """
    Class representing a data builder.

    Args:
        num_observations (int): The number of observations in the data.
    """
    def __init__(self,
                 num_observations: int,):
        self._num_observations = num_observations
        
    @abstractmethod
    def build_feature_data(self,
                           feature_value_options: Dict[str, Tuple[List[str|float] | Dict[str, List[str]], str]]):
        pass

class PropertyDataBuilder(DataBuilder):
    def __init__(self,
                 num_observations: int,
                 hierarchical_dependencies: Dict[str, str] = None):
        super().__init__(num_observations)
        self._hierarchical_dependencies = hierarchical_dependencies

    def build_feature_data(self,
                           feature_value_options: Dict[str, Tuple[List[str|float] | Dict[str, List[str]], str]]) -> pd.DataFrame:
        all_records = []
        for _ in range(self._num_observations):
            current_record = {}
            for feature_name, feature_values in feature_value_options.items():
                if self._hierarchical_dependencies.get(feature_name, None) and isinstance(feature_values[0], dict):
                    hierarchy_key = current_record[self._hierarchical_dependencies[feature_name]]
                    current_record[feature_name] = np.random.choice(feature_values[0][hierarchy_key])
                else:
                    # Check data type and generate value
                    if feature_values[1] == 'str':
                        current_record[feature_name] = np.random.choice(feature_values[0])
                    elif feature_values[1] == 'float' or feature_values[1] == 'int':
                        current_record[feature_name] = np.random.uniform(feature_values[0][0], feature_values[0][1])
                        if feature_values[1] == 'int':
                            current_record[feature_name] = int(current_record[feature_name])
                    else:
                        raise ValueError('Invalid feature data type.')
            all_records.append(current_record)
        df = pd.DataFrame(all_records)
        df['id'] = df.index
        return df
    
class BookingsDataBuilder(DataBuilder):
    def __init__(self,
                 num_observations: int,
                 property_data: pd.DataFrame,
                 start_year: int,
                 end_year: int,
                 feature_options: Dict[str, Tuple[List[str|float] | Dict[str, List[str]], str]],
                 rule_set: List[Tuple[Dict[str, str | List[float]], float]],
                 base_probability: float = 0.5):
        super().__init__(num_observations)
        self._property_data = property_data
        self._start_year = start_year
        self._end_year = end_year
        self.__time_baseline = self.__build_time_baseline()
        self.__data_baseline = self.__merge_properties_to_time_baseline()
        del self.__time_baseline # Free up memory since we have many cross joins
        self.build_feature_data(feature_options)
        self.calculate_bookings_probability(rule_set, base_probability)
        self.filter_top_probability_bookings()

    def __build_time_baseline(self) -> None:
        week_nums = pd.DataFrame({"week_num": list(range(1, 53))})
        years = pd.DataFrame({"year": list(range(self._start_year, self._end_year + 1))})
        combinations = pd.merge(years, week_nums, how='cross')

        return combinations
    
    def __merge_properties_to_time_baseline(self):
        return pd.merge(self.__time_baseline, self._property_data, how='cross')
    
    def build_feature_data(self,
                           feature_value_options: Dict[str, Tuple[List[str|float] | Dict[str, List[str]], str]]) -> None:
        data = self.__data_baseline
        num_records = data.shape[0]
        for feature_name, feature_values in feature_value_options.items():
            if feature_values[1] == 'str':
                data[feature_name] = np.random.choice(feature_values[0], num_records)
            elif feature_values[1] == 'float' or feature_values[1] == 'int':
                data[feature_name] = np.random.uniform(feature_values[0][0], feature_values[0][1], num_records)
                if feature_values[1] == 'int':
                    data[feature_name] = data[feature_name].astype(int)
            else:
                raise ValueError('Invalid feature data type.')
        self.__data_baseline = data

    def calculate_bookings_probability(self,
                                       rule_set: List[Tuple[Dict[str, str | List[float]], float]],
                                       base_probability: float = 0.5) -> None:
        data = self.__data_baseline
        data['booking_probability'] = base_probability
        for (rule, probability_adjustment) in rule_set:
            # Iterate through the rule JSON and create the pandas boolean mask
            mask = np.array([data[k] == v if isinstance(v, str) else (data[k] >= v[0]) & (data[k] <= v[1]) for k, v in rule.items()]).all(axis=0)

            # Apply the probability adjustment to the booking probability for the rows where all conditions are met
            data.loc[mask, 'booking_probability'] += probability_adjustment
        
        data['booking_probability'] = data['booking_probability'].clip(0, 1)
        self.__data_baseline = data

    def filter_top_probability_bookings(self):
        self.__data_filtered = (
            self.__data_baseline
            .sort_values('booking_probability', ascending=False)
            .head(self._num_observations)
            .drop(columns=['booking_probability'])
            .reset_index(drop=True)
            .assign(booking_id=lambda x: "B-" + x.index.astype(str))
            .rename(columns = {"id": "property_id"})
            )

    def retrieve_data(self):
        return self.__data_filtered
