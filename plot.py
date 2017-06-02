"""
Interactively plot multiple time series.

Based on:
https://matplotlib.org/devdocs/gallery/widgets/textbox.html
https://matplotlib.org/examples/widgets/slider_demo.html
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class PandasPlot(object):
    """
    Plot a Pandas DataFrame containing multiple series of data.

    Attributes:
        data (DataFrame): each column is a channel's timeseries.
        data_selection (DataFrame): selection of data for display.
        highlights (dict): identifies ranges of time in the data
            to mark for highlighting. Expected format described further
            in docstring for self.highlights_to_display()
        plot_func (func): the function used to render each channel series.
        sample_start (int): number of first sample to display
        num_display_chans (int): number of channels to display at once
        num_display_samples (int): number of samples to display at once
    """

    def __init__(self, data, highlights=None, plot_func=plt.plot,
        sample_start=0, num_display_chans=5, num_display_samps=5000):

        self.plot_func = plot_func
        self.data = data
        self.data_selection = None
        self.highlights = highlights or []
        self.sample_start = sample_start
        self.channel_start = 0
        self.num_display_chans = num_display_chans
        self.num_display_samples = num_display_samps

        self.fig, self.ax = plt.subplots()
        sns.set_style('dark')
        axcolor = 'lightgoldenrodyellow'

        ax_sample_start = plt.axes([0.10, 0.07, 0.4, 0.025], axisbg=axcolor)
        ax_ch_start = plt.axes([0.10, 0.04, 0.4, 0.025], axisbg=axcolor)

        self.slider_channel_start = DiscreteSlider(ax_ch_start,
            'Channels', 0, data.shape[1],
            increment=1, valinit=self.channel_start)
        self.slider_sample_start = DiscreteSlider(ax_sample_start,
            'Samples', 0, data.shape[0],
            increment=num_display_samps/5, valinit=self.sample_start)

        self.slider_channel_start.on_changed(self.update)
        self.slider_sample_start.on_changed(self.update)

        self.update(None)
        plt.subplots_adjust(left=0.06, right=0.98,
            top=0.96, bottom=0.15, hspace=0.03)
        plt.show()

    def highlights_to_display(self, ch_name):
        """
        Determine highlights in display range for a channel.

        This function expects that self.highlights is a
        dictionary with string channel names as keys
        and lists of integer pairs as values. The integer
        pairs represent start and end samples to highlight.

        EX)
        self.highlights =
            {'CH1': [(0, 500), (2000, 4000)],
             'CH3': [(20, 200)]}

        Args:
            ch_name (str): the channel to check for highlights.
        """
        start, end = self.data_selection.index[0], self.data_selection.index[-1]
        display_highlights = []
        if ch_name in self.highlights:
            for highlight in self.highlights[ch_name]:
                h_start, h_end = highlight
                # check whether highlight and display ranges overlap
                if h_start <= end and start <= h_end:
                    # ensure that highlight is within display selection
                    if h_start < start:
                        h_start = start
                    if h_end > end:
                        h_end = end
                    display_highlights.append((h_start, h_end))
        return display_highlights

    def plot_data_selection(self):
        """
        Plot the current data selection.
        """
        for ch_num, ch_name in enumerate(self.data_selection):
            plt.subplot(self.num_display_chans, 1, ch_num+1)
            plt.cla()
            plt.margins(x=0, y=0.1)
            # hide xticks for all but last subplot
            if ch_num+1 != self.num_display_chans:
                plt.xticks(visible=False)

            self.plot_func(self.data_selection[ch_name])
            display_highlights = self.highlights_to_display(ch_name)
            # highlight all spans of data in range,
            # and color labels for channels with highlights
            if display_highlights:
                plt.ylabel(ch_name, color='red')
                for highlight in display_highlights:
                    h_start, h_end = highlight
                    plt.axvspan(h_start, h_end, color='red', alpha=0.3)
            else:
                plt.ylabel(ch_name)

    def update(self, val):
        """
        Update variables and redraw plot when slider changes.
        """
        self.sample_start = int(self.slider_sample_start.val)
        self.channel_start = int(self.slider_channel_start.val)
        sample_end = self.sample_start + self.num_display_samples
        channel_end = self.channel_start + self.num_display_chans

        self.data_selection = self.data.iloc[self.sample_start:sample_end,
                self.channel_start:channel_end]
        self.plot_data_selection()
        self.fig.canvas.draw_idle()

class NumpyPlot(PandasPlot):
    """
    Plot a [channels x samples] Numpy matrix.

    Creates a DataFrame with the supplied matrix and
    an optional list of channel labels, then initializes
    a PandasPlot.

    Args:
        data: [channels x samples] Numpy matrix
        **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, data, **kwargs):
        if 'labels' in kwargs:
            labels = kwargs.pop('labels')
        else: # create generic list of labels
            labels = ['Ch %d' %(x) for x in range(data.shape[0])]

        data = np.transpose(data)
        data = pd.DataFrame(data, columns=labels)
        PandasPlot.__init__(self, data, **kwargs)


class DiscreteSlider(Slider):
    """
    A matplotlib slider widget with discrete steps.

    Source:
    https://stackoverflow.com/questions/13656387/can-i-make-matplotlib-sliders-more-discrete
    """

    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.5)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)

