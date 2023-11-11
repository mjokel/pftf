# import standard libraries
import configparser
from typing import List
from pathlib import Path
import ast


# TODO ------------------------------------------------------------------------
#  - TOML/YAML support
#  - tests


# class definition ------------------------------------------------------------

class Configuration:

    def __init__(self, path: str) -> None:

        self.c = None

        # check if file exists (and is not a directory)
        file_path = Path(path)
        if file_path.exists() and file_path.is_file():

            # init new ConfigParser and read file
            self.c = configparser.ConfigParser()
            self.c.read(file_path)
            self._validate()

        else:
            raise FileNotFoundError(f'File at {file_path} does not exist!')
        

    def get(self) -> configparser.ConfigParser:
        """ returns the configuration """
        return self.c
        
    
    def get_SW(self) -> List[int]:
        """ returns the smoothing windows as a integer array """
        # load SW and VW from config and parse string to integer
        # TODO: is there a better way than string literals
        return ast.literal_eval( self.c['DEFAULT']['SmoothingWindows'] )

    def get_VW(self) -> List[int]:
        """ returns the velocity windows as a integer array """
        # load SW and VW from config and parse string to integer
        # TODO: is there a better way than string literals
        return ast.literal_eval( self.c['DEFAULT']['VelocityWindows'] )


    def get_plot_output_path(self) -> Path:
        """ returns the path to store the plots at; creates directory if it doesn't exist yet """
        p = Path(f"{self.c['OUTPUT']['Path']}/plots/{self.c['DEFAULT']['Site']}-{self.c['DEFAULT']['Target']}")
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    def get_csv_output_path(self) -> Path:
        """ returns the path to store the csv file at; creates directory if it doesn't exist yet """
        p = Path(f"{self.c['OUTPUT']['Path']}/csv/{self.c['DEFAULT']['Site']}-{self.c['DEFAULT']['Target']}")
        p.mkdir(parents=True, exist_ok=True)
        return p
        
    def get_plots_enabled(self) -> bool:
        """ returns if plots should be created """
        return self.c['OUTPUT'].getboolean('PlotsEnabled')
    
    def get_csvdump_enabled(self) -> bool:
        """ returns if CSV should be dumped """
        return self.c['OUTPUT'].getboolean('CSVDumpEnabled')


    def get_logs_path(self) -> str:
        """ returns path to logs file"""
        return self.c['LOGGING']['Path']


    def _validate(self):
        """ private method for validating a given configuration """

        # check `DEFAULT - Mode`
        self._validate_Mode()
        
        # check `DEFAULT - Site`
        self._validate_notEmpty('DEFAULT', 'Site')
        
        # check `DEFAULT - Target`
        self._validate_notEmpty('DEFAULT', 'Target')
        

        # check `OUTPUT - Path` and create directory, if it doesn't exist yet
        self._validate_notEmpty('OUTPUT', 'Path')        
        
        # check `LOGGING - Path`
        self._validate_notEmpty('LOGGING', 'Path')        
        
        
        # TODO: format checks, etc.
        self._validate_notEmpty('DEFAULT', 'SmoothingWindows')        
        self._validate_notEmpty('DEFAULT', 'VelocityWindows')        


        # TODO
        # TODO
        # TODO




    # private helper methods --------------------------------------------------

    def _validate_Mode(self) -> None:
        """ raises ValueError if key `Mode` is not set to `simulation` """
        
        if ('Mode' in self.c['DEFAULT']):
            if (self.c['DEFAULT']['Mode'].lower() != "simulation"):
                raise ValueError(f'Key `Mode` must be set to `simulation`!')
        else:
            raise ValueError(f'Key `Mode` is missing!')


    def _validate_notEmpty(self, section, key) -> None:
        """ raises ValueError if `key` with string value in `section` is empty """

        if (key in self.c[section]):
            if not (self.c[section][key].strip()):
                raise ValueError(f'Key `{key}` can not be empty!')
            # else:
                # print(self.c[section][key])
        else:
            raise ValueError(f'Key `{key}` is missing!')