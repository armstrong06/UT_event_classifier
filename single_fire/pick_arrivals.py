import os
import sys
import pandas as pd
import numpy as np
import argparse
import warnings
import logging
import seisbench.models as sbm
import obspy
from obspy.core.inventory.inventory import read_inventory
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Followed this tutorial https://betterstack.com/community/guides/logging/how-to-start-logging-with-python/
# Just a simple logger for now
logger = logging.getLogger("label_phases")
stdoutHandler = logging.StreamHandler(stream=sys.stdout)
fmt = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
stdoutHandler.setFormatter(fmt)
logger.addHandler(stdoutHandler)
logger.setLevel(logging.DEBUG)

def quality_logic(sb_pick_dict, auto_time, delta_s):
    """Defines the quality of a pick. 
    {1: peak_value >= 0.5 and auto_time - peak_time <= delta_s,
     2: peak_value < 0.5 or auto_time - peak_time > delta_s,
     3: peak_value < 0.5 and auto_time - peak_time > delta_s,
     4: no deep-learning pick}

    Args:
        sb_pick (dict): seisbench pick __dict__
        auto_time (obspy UTCDateTime): arrival time of the auto pick computed 
                                        group velocity
        delta_s (float): 

    Returns:
        _type_: _description_
    """
    
    # If there is not DL pick and using auto_time, 
    # assign the worst quality 
    if sb_pick_dict is None:
        return 4
    
    # Estimate pick qual from post probs
    peak_quality = sb_pick_dict["peak_value"] >= 0.5 
    # Estimate pick qual for distance from auto pick
    delta_quality = abs(sb_pick_dict["peak_time"] - auto_time) <= delta_s

    # If both peak and delta quality are good, 
    # assign the best quality
    if peak_quality and delta_quality:
        return 1
    # If one of peak or delta quality are good, 
    # assign intermediate quality 
    elif peak_quality or delta_quality:
        return 2

    # If neither peak quality or delta quality, 
    # assign the lowest quality value for DL picks 
    return 3

def plot(st, p_at, s_at, preds, phase_picks, title, output_file_name=None):
    """ 
    Plot a station's waveform with the P and S picks predicted by a deep learning model and 
    by using the set group velocity. 

    Args:
        st (obspy Stream): contains the station waveforms
        p_at (obspy UTCDateTime): the P arrival time estimated using the group velocity 
        s_at (obspy UTCDateTime): the S arrival time estimated using the group velocity 
        preds (obspy Stream): contains the posterior probabilities from the SeisBench model
        phase_picks (seisbench PickList): pick information from preds
        title (str): the plot's title
        output_file_name (str, optional): The path to save the figure to. If None, doesn't save. Defaults to None.
    """
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, constrained_layout=True)

    ### Process the waveforms - just remove the mean and normalize ### 
    # Z component for P picks
    z_trace = st.select(channel="*Z")[0].copy().detrend(type='demean').normalize()
    # N or 2 component for the S picks
    n_trace = st.select(channel="*[2N]")[0].copy().detrend(type='demean').normalize()

    ### Plot the waveforms and group velocity predicted time ### 
    ax[0].plot(z_trace.times("matplotlib"), z_trace.data, label=z_trace.stats.channel, color="gray")
    ax[0].axvline(p_at.matplotlib_date, color="k", label="Auto P")
    ax[1].plot(n_trace.times("matplotlib"), n_trace.data, label=n_trace.stats.channel, color="gray")
    ax[1].axvline(s_at.matplotlib_date, color="k", label="Auto S")

    ### Plot the posterior probabilities from the deep learning model ### 
    for i, phase in enumerate(["P", "S"]):
        pred_trace = preds.select(channel=f"*_{phase}")[0].copy()
        #pred_trace.trim(st[0].stats.starttime+20, st[0].stats.starttime+40)
        model, pred_class = pred_trace.stats.channel.split("_")
        #print(phase, model)
        ax[i].plot(pred_trace.times("matplotlib"), pred_trace.data, label=f"{model} {pred_class}", c="C0")
        
    ### Plot the picks identified in the posterior probabilities ###
    # Also figure out the start and end time of the waveforms - these
    # are set to 10 s before the earliest P arrival time and 30 s after
    # the latest S arrival time or the start/end of the waveform
    max_time = s_at+30
    min_time = p_at-10
    for pick in phase_picks:
        pick = pick.__dict__
        i = 0
        if pick["phase"] == "S":
            i = 1

        peak_time = pick["peak_time"]
        ax[i].axvline(peak_time.matplotlib_date, c="C0")

        if peak_time > max_time:
            max_time = peak_time + 30
        if peak_time < min_time:
            min_time = peak_time - 10

    for i in range(2):
        ax[i].legend(loc="upper right")

    starttime = max(pred_trace.times("matplotlib")[0], (min_time).matplotlib_date)
    endtime = min(pred_trace.times("matplotlib")[-1], (max_time).matplotlib_date)
    ax[1].set_xlim([starttime, endtime])
    ax[1].set_ylim([-1, 1])
    # Format the date on the x-axis
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Custom date format (e.g., 2023-09-20)
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically locate the best tick positions
    plt.xticks(rotation=45)
    fig.suptitle(title)

    if output_file_name is not None:
        fig.savefig(output_file_name)
        plt.close(fig)

def format_ML_df_row(st, 
                     pick, 
                     auto_pick, 
                     delta_s):
    """Format the seisbench pick information for the output file. Columns contain information for 
       PyRocko pick file format, plus three extra columns at the end: the pick source (ML for machine learning), 
       the peak posterior probability from the seisbench model, and the arrival time difference between the 
       group velocity pick time and the ML pick time. The pick quality is defined by the quality_logic function.

    Args:
        st (obspy Stream): Station waveform and related information from mseed file
        pick (seisbench Pick): information of a single Pick estimated by the seisbench DL model   
        auto_pick (obspy UTCDateTime): arrival time of the auto (group velocity) pick
        delta_s (float): Number of seconds on either side of the auto_time to 
        look for a seisbench model pick.

    Returns:
        list: A list representing a row in the output pick file
    """
    pick = pick.__dict__
    #print(pick)
    return [f"phase: {pick['peak_time'].strftime('%Y-%m-%d')} {pick['peak_time'].strftime('%H:%M:%S.%f')[:-1]}", 
            quality_logic(pick, auto_pick, delta_s), # [2 if pick["peak_value"] < 0.5 else 1]
            f"{pick['trace_id']}{st[0].stats.location}.{st[-1].stats.channel}",
            "None",
            "None",
            "None",
            f"{pick['phase']}",
            "None",
            "False",
            "ML",
            f"{pick['peak_value']:0.2f}",
            f"{auto_pick - pick['peak_time']:2.1f}"]

def format_auto_df_row(st, auto_pick, phase):
    """Format the auto (group velocity) pick information for the output file. Columns contain information for 
       PyRocko pick file format, plus three extra columns at the end: the pick source (GV for group velocity) 
       and the peak posterior probability (None), and the arrival time difference between the 
       group velocity pick time and the ML pick time (0.0). 
       
       The pick quality is defined by the quality_logic function. 
    Args:
        st (obspy Stream): Station waveform and related information from mseed file
        auto_pick (obspy UTCDateTime): arrival time of the auto (group velocity) pick
        phase (str): phase label (P or S)

    Returns:
        list: A list representing a row in the output pick file
    """
    #print(pick)
    return [f"phase: {auto_pick.strftime('%Y-%m-%d')} {auto_pick.strftime('%H:%M:%S.%f')[:-1]}", 
            quality_logic(None, auto_pick, 0), 
            st[0].id,
            "None",
            "None",
            "None",
             phase,
            "None",
            "False",
            "GV",
            "None",
            0.0]

def make_ML_row(st, phase_picks, auto_time, delta_s=10.0):
    """Produces a single row for the output file when there are multiple picks
    identified in the seisbench model outputs for a phase type. The pick with the 
    largest posterior probability within \pm delta_s seconds of the auto arrival time 
    (group velocity) is used. If there is no seisbench pick within \pm
    delta_s seconds, then the pick with the largest posterior probability is used. 

    Args:
        st (obspy Stream): Station waveform and related information from mseed file
        phase_picks (seisbench PhaseList): pick information from seisbench model
        auto_time (obspy UTCDateTime): arrival time of the auto (group velocity) pick
        delta_s (float, optional): Number of seconds on either side of the auto_time to 
        look for a seisbench model pick. Defaults to 10.

    Returns:
        list: A list representing a row in the output pick file
    """
    pick_dict_list = []
    for pick in phase_picks:
        pick_dict_list.append(pick.__dict__)

    df = pd.DataFrame(pick_dict_list)
    delta_df = df[(df["peak_time"] > auto_time - delta_s) &
                (df["peak_time"] < auto_time + delta_s)]
    
    if len(delta_df) == 0:
        #row = make_auto_df_row(st, auto_time, phase_picks[0].__dict__["phase"])
        peak_value = df["peak_value"].max()
    else:
        peak_value = delta_df["peak_value"].max()

    selected_pick = phase_picks.select(min_confidence=peak_value)
    row = format_ML_df_row(st, selected_pick[0], auto_time, delta_s)

    return row

def write_output(df, file_path):
    """Write the pick information to an output file with the pyrocko pick file format. 
    Also adds three columns to the end: the pick source, the peak posterior probability,
    and the group velocity estimated arrival time minus the DL arrival time.
    Column values are: ["phase.AT", "QUAL", "NET.STAT.LOC.CHAN", "N1", "N2", "N3", "PHASE",
    "N4", "N5", "METHOD", "PEAK_VALUE", "AUTO_DELTA_S"], where "N?" indicates that those 
    values are just None and I don't know what they mean.
    Args:
        df (DataFrame): The rows to write to the output file. 
        file_path (str): path/file name to write the output
    """
    # Got this from ChatGPT
    # Define column widths
    col_widths = [34, 2, 16, 15, 13, 13, 9, 5, 6, 3, 5, 6]

    # Function to format each row according to column widths
    def format_row(row, col_widths):
        return ''.join(f'{str(row.iloc[i]):<{col_widths[i]}}' for i in range(len(row)))

    # Open a text file to save the DataFrame
    with open(file_path, 'w') as f:  
        # Write each row with fixed widths
        for _, row in df.iterrows():
            f.write(format_row(row, col_widths) + '\n')

    logger.info(f'DataFrame saved to {file_path}')

def main():
    ### Handle user inputs ### 
    argParser = argparse.ArgumentParser()

    # Required argument
    argParser.add_argument("-e", "--experiment", type=str, 
                           required=True, help="experiment name")
    argParser.add_argument("-s", "--source", type=str, required=True,
                           help="source type [eq, ex]")
    argParser.add_argument("-o", "--outdir", type=str, required=True,
                           help="output base directory")

    # Optional
    argParser.add_argument('--stat_list', nargs='+', type=str, 
                            help='list of station names to use. None uses all.',
                            default=None)
    argParser.add_argument('--event_list', nargs='+', type=str, 
                           help='list of event origin times to use. None uses all.',
                           default=None)
    argParser.add_argument("--av_p_vel", type=float,
                           help="average P velocity in km/s",
                           default=6.0)
    argParser.add_argument("--av_s_vel", type=float,
                           help="average S velocity in km/s",
                           default=3.39)
    argParser.add_argument("--p_thresh", type=float,
                           help="minimum posterior prob for selecting DL P picks",
                           default=0.1)
    argParser.add_argument("--s_thresh", type=float,
                           help="minimum posterior prob for selecting DL S picks",
                           default=0.1)
    argParser.add_argument("--save_plots", type=bool,
                           help="To save waveform pick plots to disk or not",
                           default=True)
    argParser.add_argument("--max_delta_s", type=float,
                           help="Seconds around the auto group velocity pick used for prioritizing DL\
                              picks and assigning quality",
                           default=15)
    # Paths 
    argParser.add_argument("--input_dir", type=str,
                        help="Path to the directory containing the Events* and Catalogs* directories",
                        default="/uufs/chpc.utah.edu/common/home/koper-group4/relu/Spectral_modeling/Utah/")
    argParser.add_argument("--cat_dir_name", type=str,
                        help="Name of the catalog directory in input_dir",
                        default="Catalogs_BASE_SSIP_073024")
    argParser.add_argument("--events_dir_name", type=str,
                            help="Name of the events directory in input_dir",
                            default="Events_BASE_SSIP_073024")
    argParser.add_argument("--out_dir_name", type=str,
                            help="Name of the output directory to create",
                            default="single_fire_picks")
    
    ### Parse the arguments
    args = argParser.parse_args()
    experiment_name = args.experiment
    source_type = args.source
    out_base_dir = args.outdir 

    stat_list = args.stat_list
    event_ot_str_list = args.event_list
    av_p_vel = args.av_p_vel
    av_s_vel = args.av_s_vel
    p_pick_thresh = args.p_thresh
    s_pick_thresh = args.s_thresh
    save_plots=args.save_plots
    auto_pick_time_delta_s = args.max_delta_s

    main_dir = args.input_dir
    cat_dir_name = args.cat_dir_name
    events_dir_name = args.events_dir_name
    out_dir_name = args.out_dir_name

    # Quick argument checks
    assert np.isin(source_type.lower(), ["ex", "eq"]), "Source type must be ex or eq"
    assert av_p_vel > av_s_vel, "P vel must be > S vel"
    assert av_p_vel > 0 and av_s_vel > 0, "Velocities must be > 0"
    assert p_pick_thresh >= 0 and p_pick_thresh <= 1.0, "P pick thresh must be between 0 and 1 (inclusive)"
    assert s_pick_thresh >= 0 and s_pick_thresh <= 1.0, "S sick thresh must be between 0 and 1 (inclusive)"
    assert auto_pick_time_delta_s >= 0, "max_delta_s must be >= 0"

    # Complete paths and read in files
    cat_file = os.path.join(main_dir, f"{cat_dir_name}/{experiment_name.upper()}/cat_{experiment_name.lower()}_{source_type.lower()}.csv")
    events_dir = os.path.join(main_dir, f"{events_dir_name}/Events_{experiment_name.upper()}_{source_type.upper()}")
    out_base_dir = os.path.join(out_base_dir, f"{out_dir_name}/{experiment_name.upper()}_{source_type.upper()}")
    logger.info(f"Catalog dir: {cat_file}")
    logger.info(f"Events dir: {events_dir}")
    logger.info(f"Output dir: {out_base_dir}")
    cat = pd.read_csv(cat_file)
    if event_ot_str_list is not None:
        cat = cat[cat.UTC.isin(event_ot_str_list)]
    logger.info(f"Using {len(cat)} events")

    # Use PhaseNet model trained with STEAD
    with warnings.catch_warnings(record=True) as w:
        # Pytorch throws a FutureWarning
        model = sbm.PhaseNet.from_pretrained("stead")

    # Column names for the dataframe
    col_names = ["phase.AT", "QUAL", "NET.STAT.LOC.CHAN", "N1", "N2", "N3", "PHASE", 
                 "N4", "N5", "METHOD", "PEAK_VALUE", "AUTO_DELTA_S"]
    
    # If the stat_list is not set, update the list for every event
    update_stat_list = False
    if stat_list is None:
        update_stat_list = True
    
    # Iterate over the event rows in the catalog
    for i, event_row in cat.iterrows():
        # Extract relavent event information
        event_ot_str = event_row["UTC"]
        event_loc = (event_row["LAT"], event_row["LON"])
        event_ot_utc = UTCDateTime(event_ot_str)
        logger.info(f"Working on: {event_ot_str}")
        # Set the paths for the waveforms and station information for this event
        wf_dir = os.path.join(events_dir, f"{event_ot_str}/Data/waveforms")
        xml_dir = os.path.join(events_dir, f"{event_ot_str}/Data/stations")
        
        inv = read_inventory(os.path.join(xml_dir, f"*xml"))
        logger.info(f"There are {len(inv)} station xml files.")
        # If the list of station was not specified, use all of them
        if update_stat_list:    
            stat_list = np.unique([chan.split(".")[1] for chan in inv.get_contents()["channels"]])

        # Set the and make the output dir for this event. Directory name is the event OT
        out_dir = os.path.join(out_base_dir, event_ot_str)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Set the pick output file name as "picks_{experiment}_{source}.txt" (e.g., "picks_base_eq.txt")
        output_file = os.path.join(out_dir, f"picks_{experiment_name.lower()}_{source_type.lower()}.txt")

        # If saving the plots to disk, make directory in event output dir called "plots"
        if save_plots:
            plot_out_dir = os.path.join(out_dir, "plots")
            if not os.path.exists(plot_out_dir):
                os.makedirs(plot_out_dir)

        # Iterate over the stations in the stat_list
        ev_rows = []
        for stat in stat_list:
            # Get the station metadata
            stat_inv = inv.select(station=stat)
            if len(stat_inv) == 0:
                logger.warning(f"No {stat} info")
                continue

            # Compute the auto P and S arrival times using the set group velocities
            stat_loc = (stat_inv[0][0].latitude, stat_inv[0][0].longitude)
            sr_dist_km = gps2dist_azimuth(stat_loc[0], stat_loc[1], event_loc[0], event_loc[1])[0]/1000
            p_at = event_ot_utc + sr_dist_km/av_p_vel
            s_at = event_ot_utc + sr_dist_km/av_s_vel

            # Read in the three component waveforms for the station
            st = obspy.read(os.path.join(wf_dir, f"*{stat}*mseed"))
            #assert len(st) == 3, f"The number of traces in the stream is {len(st)} for station {stat}. Should be 3."
            if len(st) != 3:
                logger.warning(f"There are {len(st)} traces for station {stat} (expected 3). Use with caution.")

            # Get the seisbench model posterior probabilities
            preds = model.annotate(st)
            # Get the picks corresponding to posterior probabilities greater than the pick thresholds
            phase_picks = model.classify(st, **{"P_threshold":p_pick_thresh, "S_threshold":s_pick_thresh})

            # Plot the waveforms and picks
            plot(st.copy(), p_at, s_at, preds, phase_picks.picks, 
                title=f"{experiment_name.upper()} {source_type.upper()} {event_ot_str} {stat}",
                output_file_name=[os.path.join(plot_out_dir, f"{st[0].id[:-1]}.png") if save_plots else None][0])

            # Get P pick information for the output file 
            p_picks = phase_picks.picks.select(phase="P")
            if len(p_picks) == 0:
                ev_rows.append(format_auto_df_row(st, p_at, "P"))
            else:
                ev_rows.append(make_ML_row(st, p_picks, p_at, delta_s=auto_pick_time_delta_s))

            # Get S pick information for the output file
            s_picks = phase_picks.picks.select(phase="S")
            if len(s_picks) == 0:
                ev_rows.append(format_auto_df_row(st, s_at, "S"))
            else:
                ev_rows.append(make_ML_row(st, s_picks, s_at, delta_s=auto_pick_time_delta_s))

        df = pd.DataFrame(ev_rows, 
                        columns=col_names)
        write_output(df, output_file)

if __name__ == "__main__":
    main()