# import standard libraries
import logging

# import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import local package
from reader import SimulationReader
from core import InternalState
from config import Configuration


# load configuration ----------------------------------------------------------

c = Configuration('../configs/default.ini')
# c = Configuration('../configs/no-plot.ini')


# setup logger ----------------------------------------------------------------

logging.basicConfig(filename=c.get_logs_path(), level=logging.INFO, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# read CSV file ---------------------------------------------------------------

# create reader instance
r = SimulationReader(config=c.get())


# -----------------------------------------------------------------------------

# create instance for storing internal state
s = InternalState(nrows=r.get_nrows(), SW=c.get_SW(), VW=c.get_VW())


# main forecasting loop -------------------------------------------------------

iteration = 0

while True:

    try:
        item = r.next()
        
        # add observation to internal state's displacement array
        s.arr_disp[iteration] = item[1]

        # apply smoothing and compute velocities
        s.process(iteration)


        # onset of acceleration detection -------------------------------------
        if (not s.ooa_detected):

            s.detect_ooa(iteration)


        # time of failure forecasting -----------------------------------------
        else:

            s.predict(iteration)


            # create plots, if flag in configuration is set
            if(c.get_plots_enabled()):

                s.plot_le(iteration, c.get_plot_output_path())
                s.plot_boxplot(iteration, c.get_plot_output_path())
                s.plot_ooa_criteria(iteration, c.get_plot_output_path())
                s.plot_invv(iteration, c.get_plot_output_path())
                s.plot_disp(iteration, c.get_plot_output_path())

                logger.info('Created plots.')
        

        # increment iteration counter
        iteration += 1


    # once the generator returns a `StopIteration` exception, exit the loop
    except StopIteration:

        # export the numpy array representing the internal state to CSV, if flag in configuration is set
        if(c.get_csvdump_enabled()):
            s.export(iteration, c.get_csv_output_path())
            logger.info('Exported internal state to CSV.')


        # finally, exit the loop
        logger.info('No more rows to process, terminating!')
        break 
