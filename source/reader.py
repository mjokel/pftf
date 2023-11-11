import logging
import abc
from typing import List, Any
import configparser

import pandas as pd


class AbstractReader(abc.ABC):

    """ Abstract/interface `Reader` class, actual implementations below. 

        Documentation on Abstract Base Classes (ABC)
        - https://docs.python.org/3/library/abc.html
        - https://stackoverflow.com/questions/3570796    
    """

    @abc.abstractmethod
    def next(self) -> List[Any]:
        pass


class SimulationReader(AbstractReader):

    """ Class for importing a CSV file via its file path. Provides data access 
        via generator-based method `next()`.

        
        USAGE:
        
        # import a file
        reader = SimulationReader(file_path="data.csv")

        # get next CSV row (generator!)
        reader.next()
    """

    # constructor
    def __init__(self, config: configparser.ConfigParser):

        self.file_path = config['INPUT']['File']
        self.data = None
        
        # import given CSV file
        self._import_csv(
            file_path=self.file_path, 
            dp_col=config['INPUT']['DisplacementColumn'], 
            ts_col=config['INPUT']['TimestampColumn'],
            ts_fmt=config['INPUT']['TimestampFormat'],)

        # get number of rows in CSV file
        self.nrows = len(self.data)
        
        # initialize generator on CSV data
        self.generator = self._init_generator()

    
    def _import_csv(self, file_path: str, dp_col: str, ts_col: str, ts_fmt: str) -> None:
        """ A2: private method for importing the given CSV file with pandas """
        
        self.data = pd.read_csv(
            filepath_or_buffer=file_path, 
            # TODO: sep?

            # import only these two columns
            usecols=[ts_col, dp_col], 
            
            # parse date with given format
            parse_dates=[dp_col],
            date_format=ts_fmt,
            ) 
        
        logging.info(f'imported file {self.file_path}!')


    def _init_generator(self) -> List[Any]:
        """ private method for creating the generator on CSV rows """
        for row in self.data.itertuples(index=False, name=None):
            yield row
        
    def next(self):
        """ interface to generator, yields latest CSV row """
        return next(self.generator)

    def get_nrows(self):
        """ returns the CSV file's number of rows """
        return self.nrows


    

# TODO
# class LiveReader(Reader):
