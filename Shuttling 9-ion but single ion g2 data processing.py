import os
import TimeTagger
import matplotlib
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

matplotlib.rcParams.update(
        {"font.family": "STIXGeneral",
         "xtick.labelsize": 20,
         "xtick.direction": "in",
         "xtick.major.pad": 8,
         "xtick.top": True,
         "ytick.labelsize": 20,
         "ytick.direction": "in",
         "ytick.right": True,
         "axes.labelsize": 20,
         "axes.labelpad": 10,
         "axes.grid": True
    }
)

# g2_data_path
g2_data_path = "D:/Data Storage/Multiplexing/g2_data/"

# Getting the current date and time
now = datetime.now()
# Extracting the date, hour, and minute
current_date = now.date()
current_hour = now.hour
current_minute = now.minute
# Converting to a char string
formatted_string = f"Date {current_date}, Time {current_hour}-{current_minute}"

all_raw_data = ["filewriterDate 2024-01-11, Time 19-37_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 9-56_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 10-22_9_ions.1.ttbin",
                "filewriterDate 2024-01-11, Time 18-46_9_ions.1.ttbin",
                "filewriterDate 2024-01-11, Time 18-19_9_ions.1.ttbin",
                "filewriterDate 2024-01-11, Time 17-8_9_ions.1.ttbin",
                "filewriterDate 2024-01-11, Time 16-16_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 13-34_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 14-17_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 13-56_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 16-8_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 16-27_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 17-14_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 17-35_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 18-5_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 18-30_9_ions.1.ttbin",
                "filewriterDate 2024-01-12, Time 18-48_9_ions.1.ttbin",
                "filewriterDate 2024-01-13, Time 12-27_9_ions.1.ttbin",
                "filewriterDate 2024-01-13, Time 13-5_9_ions.1.ttbin",
                "filewriterDate 2024-01-13, Time 17-12_9_ions.1.ttbin",
                "filewriterDate 2024-01-13, Time 17-39_9_ions.1.ttbin",
                "filewriterDate 2024-01-13, Time 18-24_9_ions.1.ttbin",
                "filewriterDate 2024-01-13, Time 19-29_9_ions.1.ttbin",
                "filewriterDate 2024-01-13, Time 19-47_9_ions.1.ttbin",
                "filewriterDate 2024-01-13, Time 20-36_9_ions.1.ttbin",
                "filewriterDate 2024-01-14, Time 12-42_9_ions.1.ttbin",
                "filewriterDate 2024-01-14, Time 13-10_9_ions.1.ttbin",
                "filewriterDate 2024-01-14, Time 9-38_9_ions.1.ttbin"]

# raw_data = ["filewriterDate 2024-01-11, Time 19-37_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 9-56_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 10-22_9_ions.1.ttbin",
#             "filewriterDate 2024-01-11, Time 18-46_9_ions.1.ttbin",
#             "filewriterDate 2024-01-11, Time 18-19_9_ions.1.ttbin",
#             "filewriterDate 2024-01-11, Time 17-8_9_ions.1.ttbin",
#             "filewriterDate 2024-01-11, Time 16-16_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 13-34_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 14-17_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 13-56_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 16-8_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 17-14_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 17-35_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 18-5_9_ions.1.ttbin",
#             "filewriterDate 2024-01-12, Time 18-48_9_ions.1.ttbin",
#             "filewriterDate 2024-01-13, Time 17-12_9_ions.1.ttbin"]

raw_data = ["filewriterDate 2024-01-12, Time 9-56_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 10-22_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 13-34_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 14-17_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 13-56_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 16-8_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 16-27_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 17-14_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 17-35_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 18-5_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 18-30_9_ions.1.ttbin",
            "filewriterDate 2024-01-12, Time 18-48_9_ions.1.ttbin"]

# all_raw_data = ["filewriterDate 2024-01-11, Time 19-37_9_ions.1.ttbin"]
# raw_data = ["filewriterDate 2024-01-12, Time 9-56_9_ions.1.ttbin"]

# Start_time1 = 0.0
# End_time1 = 2
# Start_time2 = 13.0
# End_time2 = 15
# Start_time3 = 25.0
# End_time3 = 27
# Start_time4 = 37.0
# End_time4 = 39
# Start_time5 = 49.0
# End_time5 = 51
# Start_time6 = 60.5
# End_time6 = 62.5
# Start_time7 = 72.0
# End_time7 = 74
# Start_time8 = 85.0
# End_time8 = 87.0
# Start_time9 = 97.5
# End_time9 = 99.5

# Time filtering window length
# time_filtering = (1 + offset) * 1e6
time_filtering = (0.3) * 1e6

offset = 0
# offset = 0

Start_time1 = 0.7
End_time1 = Start_time1 + time_filtering/1e6
Start_time2 = 13.7
End_time2 = Start_time2 + time_filtering/1e6
Start_time3 = 25.7
End_time3 = Start_time3 + time_filtering/1e6
Start_time4 = 37.7
End_time4 = Start_time4 + time_filtering/1e6
Start_time5 = 49.7
End_time5 = Start_time5 + time_filtering/1e6
Start_time6 = 61.2
End_time6 = Start_time6 + time_filtering/1e6
Start_time7 = 72.7
End_time7 = Start_time7 + time_filtering/1e6
Start_time8 = 85.2
End_time8 = Start_time8 + time_filtering/1e6
Start_time9 = 98.18
End_time9 = Start_time9 + time_filtering/1e6

# Number of bins and binwidth for correlation measurement
number = int(750 * 1.7)
binwidth_corr = 200 * 1e3

# Number of bins and binwidth for histogram measurement
number_bins = 460 * 16
binwidth_hist = 125 * 1e3 / 8

# The defination of all channels
trigger_channel = 2
PMT_channel1 = 1
PMT_channel2 = 3

# Getting the current date and time
now = datetime.now()
# Extracting the date, hour, and minute
current_date = now.date()
current_hour = now.hour
current_minute = now.minute
# Converting to a char string
formatted_string = f"Date {current_date}, Time {current_hour}-{current_minute}"

# The Virtual Time Tagger is initialized similar to the physical one
tagger = TimeTagger.createTimeTaggerVirtual()
fig = plt.figure()

final_result = np.zeros(number)
final_hist = np.zeros(number_bins)

# The loop allows us to repeat the replay several times
for data in raw_data:
    binwidth = binwidth_corr
    n_bins = number   # Adjust number of bins as needed

    open_gate1 = TimeTagger.DelayedChannel(tagger=tagger, input_channel=trigger_channel, delay=Start_time1 * 1e6)
    open_gate_channel1 = open_gate1.getChannel()
    open_gate2 = TimeTagger.DelayedChannel(tagger=tagger, input_channel=trigger_channel, delay=Start_time2 * 1e6)
    open_gate_channel2 = open_gate2.getChannel()
    open_gate3 = TimeTagger.DelayedChannel(tagger=tagger, input_channel=trigger_channel, delay=Start_time3 * 1e6)
    open_gate_channel3 = open_gate3.getChannel()
    open_gate4 = TimeTagger.DelayedChannel(tagger=tagger, input_channel=trigger_channel, delay=Start_time4 * 1e6)
    open_gate_channel4 = open_gate4.getChannel()
    open_gate5 = TimeTagger.DelayedChannel(tagger=tagger, input_channel=trigger_channel, delay=Start_time5 * 1e6)
    open_gate_channel5 = open_gate5.getChannel()
    open_gate6 = TimeTagger.DelayedChannel(tagger=tagger, input_channel=trigger_channel, delay=Start_time6 * 1e6)
    open_gate_channel6 = open_gate6.getChannel()
    open_gate7 = TimeTagger.DelayedChannel(tagger=tagger, input_channel=trigger_channel, delay=Start_time7 * 1e6)
    open_gate_channel7 = open_gate7.getChannel()
    open_gate8 = TimeTagger.DelayedChannel(tagger=tagger, input_channel=trigger_channel, delay=Start_time8 * 1e6)
    open_gate_channel8 = open_gate8.getChannel()
    open_gate9 = TimeTagger.DelayedChannel(tagger=tagger, input_channel=trigger_channel, delay=Start_time9 * 1e6)
    open_gate_channel9 = open_gate9.getChannel()

    open_gate = TimeTagger.Combiner(tagger=tagger, channels=[open_gate_channel1,
                                                             open_gate_channel2,
                                                             open_gate_channel3,
                                                             open_gate_channel4,
                                                             open_gate_channel5,
                                                             open_gate_channel6,
                                                             open_gate_channel7,
                                                             open_gate_channel8,
                                                             open_gate_channel9])
    open_gate_channel = open_gate.getChannel()

    close_gate = TimeTagger.DelayedChannel(tagger=tagger, input_channel=open_gate_channel, delay=time_filtering)
    close_gate_channel = close_gate.getChannel()

    gated_PMT_channel1 = TimeTagger.GatedChannel(tagger=tagger, input_channel=PMT_channel1, gate_start_channel=open_gate_channel, gate_stop_channel=close_gate_channel)
    gated_PMT_channel2 = TimeTagger.GatedChannel(tagger=tagger, input_channel=PMT_channel2, gate_start_channel=open_gate_channel, gate_stop_channel=close_gate_channel)
    
    hist = TimeTagger.Histogram(tagger=tagger, click_channel=PMT_channel1, start_channel = trigger_channel, binwidth = binwidth_hist, n_bins = number_bins)
    # hist = TimeTagger.Histogram(tagger=tagger, click_channel=gated_PMT_channel2.getChannel(), start_channel = trigger_channel, binwidth = binwidth_hist, n_bins = number_bins)

    # corr = TimeTagger.Correlation(tagger=tagger, channel_1=gated_PMT_channel1.getChannel(), channel_2=gated_PMT_channel2.getChannel(), binwidth=binwidth, n_bins=n_bins)
    corr = TimeTagger.Correlation(tagger=tagger, channel_1=gated_PMT_channel1.getChannel(), channel_2=gated_PMT_channel2.getChannel(), binwidth=binwidth_corr, n_bins=n_bins)
    # measurement_time = int(0.001 * 3600 * 1e12)  # Adjust measurement time as needed

    # In general, the data are replayed at the same speed as they have been acquired. You can modify the speed by a factor or replay them as fast es possible by speed < 0
    tagger.setReplaySpeed(speed=-1)

    # Start the replay
    filename = g2_data_path + data
    tagger.replay(file=filename)

    while not tagger.waitForCompletion(timeout=-1):
        fig.clear()
        plt.plot(corr.getIndex(), corr.getData())
        plt.xlabel("Time difference [ps]")
        plt.ylabel("Counts")
        plt.pause(.1)

    # Extracting the date, hour, and minute
    current_date = now.date()
    current_hour = now.hour
    current_minute = now.minute
    # Converting to a char string
    formatted_string = f"Date {current_date}, Time {current_hour}-{current_minute}"
    print(formatted_string)

    # Get the correlation data
    hist_data = hist.getData()
    hist_index = hist.getIndex() / 1e6
    final_hist += hist_data
    # np.save('hist_meas_'+formatted_string+'.npy', hist_index, hist_data)
    
    g2_data = corr.getData()
    index = corr.getIndex() / 1e6
    final_result += g2_data
    # np.save('corr_meas_'+formatted_string+'.npy', index, g2_data)

    # Getting the current date and time
    now = datetime.now()

    # np.save('corr.getData_'+formatted_string+'.npy', corr.getData())
    # np.save('corr.getIndex_'+formatted_string+'.npy', corr.getIndex())

# Free the TimeTagger
TimeTagger.freeTimeTagger(tagger)

# Plot the g(2) function
# np.save('data_final_corr_meas_'+formatted_string+'.npy', final_result)
# np.save('index_final_corr_meas_'+formatted_string+'.npy', index)

np.save('final_result_shuttling_9_ion_chain_1.npy', final_result)
np.save('final_result_shuttling_9_ion_chain_index_1.npy', index)

plt.figure(figsize = (10, 5))
plt.plot(index, final_result)
plt.title("9 ions coincidence counts", fontdict={'size':20})
plt.xlabel("Delay (us)")
plt.ylabel("Counts")
plt.show()