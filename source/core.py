import logging
import math
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


class InternalState:

    # NOTE: https://stackoverflow.com/questions/13784192
    #  -> "NEVER grow a DataFrame row-wise!"
    #  -> hence, we go with Numpy arrays, as they can be preallocated, 
    #     instead of Pandas DataFrames that don't allow this

    def __init__(self, nrows: int, SW: list[int], VW: list[int]) -> None:

        # main goal: avoid use of global variables

        self.SW = SW
        self.VW = VW

        self.ooa_detected = False
        self.ooa_index = None

        # preallocate array for displacement values
        self.arr_disp = np.zeros(nrows, dtype=float)

        # preallocate array for smoothed displacement values
        self.arr_smoo = np.zeros((nrows, len(SW)), dtype=float)


        # preallocate array for velocities
        self.arr_velo = np.zeros((nrows, len(SW), len(VW)), dtype=float)

        # preallocate array for inverse velocity quantiles
        self.arr_quan = np.zeros((nrows, len(SW), len(VW), 2), dtype=float)


        # preallocate boolean array for onset of acceleration detection
        self.arr_ooa = np.zeros((nrows, len(SW), len(VW)), dtype=bool)

        # preallocate array for 'time of failure' and 'time to failure' forecasts
        self.arr_fore = np.zeros((nrows, len(SW), len(VW)+4, 2 ), dtype=float)



    def process(self, iteration: int) -> None:

        # NOTE
        # 
        # when inserting values in the current iteration:
        #  - we use `iteration` as index: array[iteration] = 3.1415 
        #
        # when accessing previously computed rows in arrays:
        #  - syntax: array[start:stop-1, ...]
        #  - we thus use `1+iteration` as stop value to obtain all values until the current iteration
        #  - to obtain the last sw values, we can use `array[ (1 + iteration - sw):(1 + iteration) ] `,
        #    however, if `start<0`, then `[]` is returned!


        # smooth displacement value according to smoothing windows ----------
        for i, sw in enumerate(self.SW):

            # get last `sw` observations from arr_disp: items from `start` through `stop-1`;
            # if `start < 0`, then `[]` is returned
            x = self.arr_disp[ (1 + iteration - sw):(1 + iteration) ] 

            # smooth displacement over last sw observations by computing the mean and insert into arr_smoo;
            # note: helper array x must contain at least sw values
            if (len(x) >= sw):
                self.arr_smoo[iteration, i] = np.mean(x)


            # compute velocities ----------
            for j, vw in enumerate(self.VW):
                
                # get last `vw` smoothed displacement observations from arr_smoo via `i`
                vals_y = self.arr_smoo[ (1 + iteration - vw):(1 + iteration), i ]
                
                # remove NAN values from array; ~ corresponds to logical negation;
                # https://stackoverflow.com/questions/11620914
                vals_y = vals_y[~np.isnan(vals_y)]
                
                # ensure there are enough values to fit a model;
                # the else clause is not really necessary, as value in arr_velo is 0 by default
                if( len(vals_y) == vw ):

                    #print(iteration, i, j, vals_y)
        
                    vals_x = np.arange(0, vw) # values are generated within the half-open interval [start, stop)
                    
                    # fit linear model of form `m*x+t` to data and get slope m and intersect t
                    (m, t) = np.polyfit(vals_x, vals_y, 1)
                    
                    
                    # if slope is greater than 0, assign `1/m` i.e. inverse model slope to velocity array
                    if (m > 0):
                        self.arr_velo[iteration, i, j] = 1/m
                        
                        
                        
                    # compute quantiles ----------
                    
                    # get all (i.e. start-component is 0) available observations of the current iteration-vw-sw combination
                    vals_q = self.arr_velo[ 0:(1 + iteration), i, j]
                    
                    # compute 1% and 50% quantiles and assign to array;
                    # for that, exclude `0` observations as they negatively affect the quantile computation; 
                    # also, ensure helper array vals_q contains at least one element;
                    vals_q = vals_q[vals_q > 0]

                    if (len(vals_q) > 0):    
                        (q1, q50) = np.quantile(vals_q, q = [0.01, 0.5])
                        self.arr_quan[iteration, i, j, 0] = q1
                        self.arr_quan[iteration, i, j, 1] = q50 


    def detect_ooa(self, iteration: int) -> None:

        # NOTE: with this check, OOA is ONLY DETECTED ONCE ie. no follow-up detections!
        
        for i, sw in enumerate(self.SW):
            for j, vw in enumerate(self.VW):

                c1 = (self.arr_smoo[iteration, i] - self.arr_smoo[iteration-math.floor(0.5*vw), i]) > (self.arr_smoo[iteration-math.floor(0.5*vw), i] - self.arr_smoo[iteration-vw, i])        
                c2 = self.arr_velo[iteration, i, j] < self.arr_velo[(iteration-vw), i, j]
                c3 = self.arr_quan[iteration, i, j, 1] < self.arr_quan[(iteration-vw), i, j, 1]
                c4 = self.arr_velo[iteration, i, j] < self.arr_quan[iteration, i, j, 0]

                # combine criteria to single value and assign to array
                self.arr_ooa[iteration, i, j] = all([c1, c2, c3, c4])

                # if last `sw` observations are True, then OOA is detected
                vals_ooa = self.arr_ooa[(1 + iteration - vw):(1 + iteration), i, j]
                if( (len(vals_ooa) == vw) and all(vals_ooa) ):

                    # TODO: 
                    logging.info(f'{iteration} {i} {j}: OOA detected at index {iteration-vw}!')
                    # print(f'{iteration} {i} {j}: OOA detected at index {iteration-vw}!')

                    self.ooa_detected = True
                    self.ooa_index = iteration-vw


    def predict(self, iteration:int) -> None:

        for i, sw in enumerate(self.SW):
            for j, vw in enumerate(self.VW):

                vals_y = self.arr_velo[self.ooa_index:(1+iteration), i, j]
                vals_x = np.arange(self.ooa_index, (1+iteration))
                
                if( (len(vals_y) == len(vals_x)) and (len(vals_y) > 0) ):

                    # fit linear model of form `m*x+t` to data and get slope m and intersect t
                    (m, t) = np.polyfit(vals_x, vals_y, 1)

                    # x-intercept:  y = mx+t for y=0  <=>  -t/m = x
                    x_intercept = -t/m

                    
                    # NOTE: based on the assumption that the data points are sampled at equidistant
                    #       time steps, we can simply use the values row indices as x coordinates when
                    #       fitting the linear model;
                    #       in order to obtain temporal failure time predictions, we can multiply the TTF
                    #       with the time span between two equidistant points (sampling frequency?)

                    # NOTE: while this approach wastes the array's capacity before row `iteration`,
                    #       implementing another counter and working with offsets would simply mirror
                    #       the issue, leading to the capacity on the array's other end being wasted
                    
                    # save `time of failure`
                    self.arr_fore[iteration, i, j, 0] = x_intercept
                    
                    # compute and save `time to failure`
                    self.arr_fore[iteration, i, j, 1] = x_intercept - iteration
            
            
            # note:  from START:STOP-1
            # hence: new value at STOP
            #        where STOP = len(VW)
            
            # compute time of failure
            forecasts = self.arr_fore[iteration, i, 0:len(self.VW), 0]
            self.arr_fore[iteration, i, (len(self.VW)+0), 0] = forecasts.mean() 
            self.arr_fore[iteration, i, (len(self.VW)+1), 0] = forecasts.min()
            self.arr_fore[iteration, i, (len(self.VW)+2), 0] = forecasts.max()
            self.arr_fore[iteration, i, (len(self.VW)+3), 0] = forecasts.max() - forecasts.min()
            
            #print(f'{iteration} span: {arr_fore[iteration, i, (len(VW)+1), 0]}')
            
            
            # compute time to failure: subtract "time when forecast was made" from forecast
            forecasts = forecasts - iteration
            self.arr_fore[iteration, i, (len(self.VW)+0), 1] = forecasts.mean() 
            self.arr_fore[iteration, i, (len(self.VW)+1), 1] = forecasts.min()
            self.arr_fore[iteration, i, (len(self.VW)+2), 1] = forecasts.max()
            self.arr_fore[iteration, i, (len(self.VW)+3), 1] = forecasts.max() - forecasts.min()
            
            # TODO
            #print(f'{iteration} span: {arr_fore[iteration, i, (len(VW)+1), 1]}')
            # print(f'{iteration} TTF: {np.around(self.arr_fore[iteration, i, (len(self.VW)+0), 1], 2)}')
            
            print(f'{iteration}-{sw} TTF: {np.around(self.arr_fore[iteration, i, (len(self.VW)+0), 1], 2)}')



    def plot_le(self, iteration: int, path: str)  -> None:
        """ matplotlib version of life expectancy with failure window for a single smoothing window """

        # iterate smoothing windows (NOTE: smoothing window index and smoothing window)
        for swi, sw in enumerate(self.SW):
            
            # we are interested in ...
            #  · all rows since OOA detection  → arg1 = `ooa_index:`
            #  · for this EXAMPLE a SINGLE SW  → arg1 = `0`, yet we need to do this for EACH smoothing window!
            #  · VW summary statistics         → arg3 = `len(VW):len(VW)+4`
            #  · time TO failure               → arg4 = `1`
            #
            # summary statistics indices: +0: mean; +1: min; +2: max; +3: span
            data = self.arr_fore[self.ooa_index:iteration, swi, len(self.VW):len(self.VW)+4, 1]

            # get number of iterations ie. observations in `data`;
            # yes, this is a little confusing...
            observations_count = len(data)

            
            fw_data = data
            
            # compute failure window
            fw_min = fw_data[:, 1]-0.5*fw_data[:, 3]
            fw_max = fw_data[:, 2]+0.5*fw_data[:, 3]
            
            # get x values for plot: indices/timestamp starting at OOA, up to the last observation
            x_vals = np.arange(self.ooa_index, self.ooa_index+observations_count)

            
            # create Matplotlib figure and axes objects
            fig, ax = plt.subplots()
            
            
            # failure window
            ax.fill_between(x_vals, fw_min, fw_max, color='gray', alpha=.2, linewidth=0, label='failure window')
            
            # forecasts per individual velocities
            for i, vw in enumerate(self.VW):
                y_vals = self.arr_fore[self.ooa_index:iteration, swi, i, 1]
                ax.plot(x_vals, y_vals, label=f'vw_{vw}h')
                
            # mean across forecasts
            ax.plot(x_vals, fw_data[:, 0], marker='o', color='black', label='mean forecast')
            
            # set axis labels, plot legend and title
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=5, frameon=False, fancybox=False)
            ax.set_xlabel('date')
            ax.set_ylabel('life expectancy [h]')
            ax.set_title(f'Life Expectancy Forecast for SW={sw}')

            # export to disk
            fig.savefig(fname=f"{path}/sw{sw}_LE-iteration-{iteration}.pdf", bbox_inches='tight')
            plt.close(fig)
        
        

    def plot_boxplot(self, iteration: int, path: str) -> None:
        """ tof box plot across all vw of a single smoothing window, based on Matplotlib """

        # iterate smoothing windows (NOTE: smoothing window index and smoothing window)
        for swi, sw in enumerate(self.SW):

            # we are interested in ...
            #  · all rows since OOA detection  → arg1 = `ooa_index:`
            #  · for this EXAMPLE a SINGLE SW  → arg1 = `0`, yet we need to do this for EACH smoothing window!
            #  · all smoothing windows         → arg3 = `0:len(VW)`
            #  · time OF failure               → arg4 = `0`
            data = self.arr_fore[self.ooa_index:iteration, swi, 0:len(self.VW), 0]

            # print("iteration = ", iteration, "---------------")
            # print(data)

            # index of latest timestamp; must be derived before the subsequent operations
            latest_timestamp = self.ooa_index+len(data)


            # exclude last row from data frame: used separately for overplotting!
            last_row = data[-1] # last row only
            data = data[:-1]    # all elements except for the last one

            # NOTE: `data` is a 2d array, where each "inner array" represents a table row
            #       → by transposing (`.T`), each "inner array" becomes a table column
            #
            # NOTE: `plt.boxplot()` supports grouped box plots if a list of arrays is provided; 
            #       however, it will ignore any list element containing NANs
            #       → hence, we need to filter the NANs per column
            #
            # METHOD:
            # use list comprehension to filter out NANs in each column of the transposed "input" aaa;
            # returns a list of arrays/columns (might be of different lengths) without NANs
            # data = [column[~np.isnan(column)] for column in data.T] # for NAN
            
            
            data = [sublist for sublist in data.T if any(sublist)]
            if len(data) == 0:
                return False

            
            # get data for computing the failure window
            # index explanation: +0: mean; +1: min; +2: max; +3: span
            fw_data = self.arr_fore[self.ooa_index:iteration, 0, len(self.VW)+1:len(self.VW)+4, 0]
            
            # only the latest failure window is relevant, ie. drop everything except the last row
            fw_data = fw_data[-1]
            
            # compute failure window
            fw_min = fw_data[0]-0.5*fw_data[2]
            fw_max = fw_data[1]+0.5*fw_data[2]

                
            # create Matplotlib figure and axes objects
            fig, ax = plt.subplots()
            
            
            # add horizontal span representing the failure window 
            ax.axhspan(ymin=fw_min, ymax=fw_max, color='gray', alpha=.2, linewidth=0, label='failure window')

            
            # create grouped boxplot
            # ax.boxplot(data)
            ax.boxplot(data, labels=[f'{x}h' for x in self.VW])

            # add latest forecasts as red diamonds
            # NOTE: `x` is just a filler to form a list from [1, ..., len(last_row)+1]
            ax.scatter(x=np.arange(len(last_row))+1, y=last_row, color='red', marker="D", label='latest forecast')


            # add horizontal line representing current time stamp
            ax.axhline(y=latest_timestamp, color='black', label='latest timestamp')

            # format abscissa as date(time)
            #ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
            #ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
            #ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))

            # set axis labels, plot legend and title
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=3, frameon=False, fancybox=False)
            ax.set_xlabel('velocity window [h]')
            ax.set_ylabel('date')
            ax.set_title(f'Velocity Window Boxplots of SW={sw}')

            # export to disk
            fig.savefig(fname=f"{path}/sw{sw}_BX-iteration-{iteration}.pdf", bbox_inches='tight')
            plt.close(fig)
        


    def plot_ooa_criteria(self, iteration: int, path: str) -> None:
        """ plot of OOA detection across all smoothing windows """
        
        # TODO: check for OOA index, there may be an issue!

        # iterate smoothing windows (NOTE: smoothing window index and smoothing window)
        for swi, sw in enumerate(self.SW):


            # TODO: remove hard-coded reference!
            # include some pre-OOA observations
            pre = -3

            # get "combined criterion" (=C1 & C2 & C3 & C4) for each velocity window of smoothing window `sw`
            #  · all rows since OOA detection  → arg1 = `ooa_index:`
            #  · for this EXAMPLE a SINGLE SW  → arg1 = `0`, yet we need to do this for EACH smoothing window!  
            #  · all velocity windows          → arg3 = `0:len(VW)`
            #data = arr_ooa[ooa_index:, swi, 0:len(VW)]
            data = self.arr_ooa[pre+self.ooa_index:iteration, swi, 0:len(self.VW)] # include some pre-OOA observations
            
            
            # get number of iterations ie. observations in `data`;
            # yes, this is a little confusing...
            observations_count = len(data)
            
            # transpose `data`
            data = data.T
            
            # get x values for plot: indices/timestamp starting at OOA, up to the last observation
            #x_vals = np.arange(ooa_index, ooa_index+observations_count)
            x_vals = np.arange(pre+self.ooa_index, pre+self.ooa_index+observations_count)

            
            # create Matplotlib figure and axes objects
            fig, ax = plt.subplots()
            
            # hightlight OOA with vertical line
            ax.axvline(x=self.ooa_index, color='darkgray', linestyle='dashed', label='OOA')

            # for each velocity window, add the corresponding data points
            for i, val in enumerate(self.VW):
                
                # get indices of `True` observations
                x_vals_true = x_vals[data[i]]
                y_vals_true = data[i][data[i]] * i # arrange observation values along y-axis by multiplying with the index of the corresponding smoothing window
                
                # get indices of `False` observations
                x_vals_false = x_vals[~data[i]]
                y_vals_false = ( data[i][~data[i]] + 1 ) * i # add 1, such that the multiplication works (False==0); also, see above!
                
                # True observations: White Rectangle
                # NOTE: the 'label=...' argument ensures only one marker per category in the plot legend
                # ax.scatter(x_vals_true, y_vals_true, marker='s', color='white', s=100, label=f'True' if i == 1 else "") 
                ax.scatter(x_vals_true, y_vals_true, marker='s', color='green', s=100, label=f'True' if i == 1 else "") 
        
                # False observations: Black Triangle
                ax.scatter(x_vals_false, y_vals_false, marker='s', color='black', s=100, label=f'False' if i == 1 else "")
                
            # set axis labels, plot legend and title
            ax.set_xticks(x_vals)
            ax.set_yticks([i for i in range(len(self.VW))]) # create list of integers from 0 to len(SW)
            ax.set_yticklabels([f'{x}h' for x in self.VW])  # set custom y-axis labels
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize='small', ncol=3, frameon=False, fancybox=False)
            ax.set_xlabel('date')
            ax.set_ylabel('velocity windows [h]')
            ax.set_title(f'OOA Detection Criteria across velocity windows of SW={sw}')
            
            # export to disk
            fig.savefig(fname=f"{path}/sw{sw}_OC-iteration-{iteration}.pdf", bbox_inches='tight')
            plt.close(fig)
    


    def plot_invv(self, iteration: int, path: str) -> None:

        """ returns line plot with all velocities per single smoothing window """

        # iterate smoothing windows (NOTE: smoothing window index and smoothing window)
        for swi, sw in enumerate(self.SW):

            # include some pre-OOA observations
            pre = -3
            
            # we are interested in ...
            #  · all rows since OOA detection  → arg1 = `ooa_index:`
            #  · for this EXAMPLE a SINGLE SW  → arg1 = `0`, yet we need to do this for EACH smoothing window!
            #  · all velocity windows          → arg3 = `:``
            #ys = arr_velo[ooa_index:, swi, :]
            ys = self.arr_velo[pre+self.ooa_index:iteration, swi, :] # include some pre-OOA observations
            
            
            # get x values for plot: indices/timestamp starting at OOA, up to the last observation
            #x_vals = np.arange(ooa_index, ooa_index+len(ys))
            x_vals = np.arange(pre+self.ooa_index, pre+self.ooa_index+len(ys))
            #display(x_vals)
            
            
            # create Matplotlib figure and axes objects
            fig, ax = plt.subplots()
            
            # highlight OOA with vertical line
            ax.axvline(x=self.ooa_index, color='darkgray', linestyle='dashed', label='OOA')
            
            # add individual velocities
            for i, vw in enumerate(self.VW):
                # NOTE: `ys[:, i]` obtains the rows in column i of 2D array `ys` as a 1D array
                # display(ys[:, 0])
                ax.plot(x_vals, ys[:, i], label=f'vw_{vw}h')
            
            
            # set axis labels, plot legend and title
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=4, frameon=False, fancybox=False)
            ax.set_xlabel('date')
            ax.set_ylabel('inverse velocity [h/mm]')
            ax.set_title(f'Inverse Velocities for SW={sw}')
            
            # export to disk
            fig.savefig(fname=f"{path}/sw{sw}_IV-iteration-{iteration}.pdf", bbox_inches='tight')
            plt.close(fig)
    


    def plot_disp(self, iteration: int, path: str) -> None:

        """ returns line plot of the displacement since OOA """
        
        # include some pre-OOA observations
        pre = -3
        
        ys = self.arr_disp[pre+self.ooa_index:iteration] # include some pre-OOA observations
        #ys = arr_disp[ooa_index:]
        
        # get x values for plot: indices/timestamp starting at OOA, up to the last observation
        x_vals = np.arange(pre+self.ooa_index, pre+self.ooa_index+len(ys))
        
        
        # create Matplotlib figure and axes objects
        fig, ax = plt.subplots()
        
        
        # hightlight OOA with vertical line
        ax.axvline(x=self.ooa_index, color='darkgray', linestyle='dashed', label='OOA')
        #ax.text(ooa_index + 0.1, 10, 'My Label') → doesn't work as intended (ie like in ggplot)
        
        # displacement
        ax.plot(x_vals, ys, color='black', label='displacement')
        
        # set axis labels, plot legend and title
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=2, frameon=False, fancybox=False)
        ax.set_xlabel('date')
        ax.set_ylabel('displacement [mm]')
        ax.set_title('Displacement since OOA')

        # export to disk
        fig.savefig(fname=f"{path}/DP-iteration-{iteration}.pdf", bbox_inches='tight')
        plt.close(fig)
    


    def export(self, iteration: int, base_path: Path) -> None:
        """ exports displacement, smoothed displacement, velocities, quantiles, 
            OOA detection and forecasting statistics per smoothing window to CSV 
        """

        # iterate smoothing windows ----------
        for i, sw in enumerate(self.SW):

            # init list with CSV headers for iteration, displacement and smoothed displacement
            h1 = ['iter', 'disp', f'smoo{sw}']

            # CSV headers for each velocity window
            h2 = [f'vw{x}' for x in self.VW]

            # CSV headers for each velocity quantile
            h3 = [f'vw{x}_{q}' for x in self.VW for q in ['q1', 'q50']]

            # CSV headers for OOA criteria per velocity window
            h4 = [f'vw{x}_ooa' for x in self.VW]

            # CSV headers for 'time to failure' and 'time of failure' forecasts
            h5a = [f'vw{x}_tof' for x in self.VW]
            h5b = [f'tof_{x}' for x in ['mean', 'min', 'max', 'span']]
            h5c = [f'vw{x}_ttf' for x in self.VW]
            h5d = [f'ttf_{x}' for x in ['mean', 'min', 'max', 'span']]
            h5 = h5a + h5b + h5c + h5d

            # concatenate lists, create string by joining with `,` and add line break
            h = ','.join((h1 + h2 + h3 + h4 + h5)) + '\n'

            # logging.info(f'CSV header names: {h}')
            
            # create CSV file and write header row
            with open(f'{base_path}/sw{sw}.csv', 'w', newline='') as file:
                file.write(h)


        # iterate smoothing windows ----------
        for i, sw in enumerate(self.SW):

            # iterate array rows ----------
            for iter in range(0, iteration): # TODO
            # for iter in range(0, 20):

                # TODO: restructure as this is inefficient!
                # TODO: tests!

                # gather iteration data (of current smoothing window) and write to file
                d1a = iter

                # raw and smoothed displacement
                d1b = self.arr_disp[iter]
                d1c = self.arr_smoo[iter, i]

                # create array from the above to allow for concatenating later
                # TODO: casts `d1a` (iteration, INT) to FLOAT!
                d1 = np.array([d1a, d1b, d1c])

                # velocities
                d2 = self.arr_velo[iter, i, :]

                # velocity quantiles
                d3 = np.concatenate(self.arr_quan[iter, i, :, :])

                # OOA criteria
                # TODO: concatenating casts `d4` (OOA criteria, BOOL) to FLOAT!
                d4 = self.arr_ooa[iter, i, :]

                # 'time to failure' and 'time of failure' and respective summary statistics
                d5a = self.arr_fore[iter, i, :, 0]
                d5b = self.arr_fore[iter, i, :, 1]


                # NOTE: np.concatenate expects a tuple! see: https://stackoverflow.com/a/62956993
                d = np.concatenate((d1,d2,d3,d4, d5a,d5b))


                # open file in 'append' mode
                with open(f'{base_path}/sw{sw}.csv', 'a') as fp:
                    csv_writer = csv.writer(fp)

                    # NOTE: expects a single iterable!
                    csv_writer.writerow(d) 
