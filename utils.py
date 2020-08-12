# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython import display
import matplotlib.pyplot as plt
import time

class ParkingWorld:
    
    def __init__(self,
                 num_spaces=10,
                 num_prices=4,
                 price_factor=0.1,
                 occupants_factor=1.0,
                 null_factor=1 / 3):
        self.num_spaces = num_spaces
        self.num_prices = num_prices
        self.occupants_factor = occupants_factor
        self.price_factor = price_factor
        self.null_factor = null_factor
        self.State = [num_occupied for num_occupied in range(num_spaces + 1)]
        self.Action = list(range(num_prices))
    
    def transitions(self, s, a):
        return np.array([[r, self.probability(s_, r, s, a)] for s_, r in self.support(s, a)])
    
    def support(self, s, a):
        return [(s_, self.reward(s, s_)) for s_ in self.State]
    
    def reward(self, s, s_):
        return self.state_reward(s) + self.state_reward(s_)
    
    def state_reward(self, s):
        if s == self.num_spaces:
            return self.null_factor * s * self.occupants_factor
        else:
            return s * self.occupants_factor
    
    def probability(self, s_, r, s, a):
        if r != self.reward(s, s_):
            return 0
        else:
            center = (1 - self.price_factor
                      ) * s + self.price_factor * self.num_spaces * (
                          1 - a / self.num_prices)
            emphasis = np.exp(-abs(np.arange(2 * self.num_spaces) - center) / 5)
            if s_ == self.num_spaces:
                return sum(emphasis[s_:]) / sum(emphasis)
            return emphasis[s_] / sum(emphasis)

def plot(V, pi):
    # plot value
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.axis('on')
    ax1.cla()
    states = np.arange(V.shape[0])
    ax1.bar(states, V, edgecolor='none')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Value', rotation='horizontal', ha='right')
    ax1.set_title('Value Function')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax1.yaxis.grid()
    ax1.set_ylim(bottom=V.min())
    # plot policy
    ax2.axis('on')
    ax2.cla()
    im = ax2.imshow(pi.T, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    ax2.invert_yaxis()
    ax2.set_xlabel('State')
    ax2.set_ylabel('Action', rotation='horizontal', ha='right')
    ax2.set_title('Policy')
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.arange(start, end), minor=True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end), minor=True)
    ax2.grid(which='minor')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.20)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Probability', rotation=0, ha='left')
    fig.subplots_adjust(wspace=0.5)
    display.clear_output(wait=True)
    display.display(fig)
    time.sleep(0.001)
    plt.close()