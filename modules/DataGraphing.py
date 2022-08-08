import os, sys
PERCEPT_DIR = os.getenv('PERCEPT_DIR')
sys.path.append(PERCEPT_DIR)

import numpy as np
import pandas as pd
import copy
from datetime import datetime, timezone
import dateutil

from scipy import signal, stats, optimize
import Percept
import SignalProcessingUtility
import PythonUtility

import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import dates

def addShadedErrorBar(fig, x, y, stderr, color="#3BDEFF", alpha=0.5, row=None, col=None, legendgroup=None):
    RGB = [int(color[i:i+2], 16) for i in (1, 3, 5)]

    fig.add_trace(
                    go.Scatter(x=x,
                               y=y+stderr,
                               mode="lines",
                               line=dict(color=color, width=0.0),
                               fill=None,
                               hoverinfo="skip",
                               showlegend=False,
                               legendgroup=legendgroup),
                    row=row, col=col
                )

    fig.add_trace(
                    go.Scatter(x=x,
                               y=y-stderr,
                               mode="lines",
                               line=dict(color=color, width=0.0),
                               fill="tonexty", fillcolor="rgba({0},{1},{2},{3})".format(RGB[0], RGB[1], RGB[2], alpha),
                               hoverinfo="skip",
                               showlegend=False,
                               legendgroup=legendgroup),
                    row=row, col=col
                )
    return fig

def colorTextFromCmap(color):
    if type(color) == str:
        colorInfoString = color.split(",")
        colorInfoString = [string.replace("rgb(","").replace(")","") for string in colorInfoString]
        colorInfo = [int(i) for i in colorInfoString]
    else:
        colorInfo = np.array(color[:-1]) * 255
    colorText = f"#{hex(int(colorInfo[0])).replace('0x',''):0>2}{hex(int(colorInfo[1])).replace('0x',''):0>2}{hex(int(colorInfo[2])).replace('0x',''):0>2}"
    return colorText

def getBaselinePSD(Data, index):
    if len(Data["SurveyPSD"]) == 0:
        return "", [""]

    [b,a] = signal.butter(5, np.array([1,100])*2/250, 'bp', output='ba')
    Timestamp = list()
    for Stream in Data["SurveyPSD"]:
        filtered = signal.filtfilt(b, a, Stream["Data"])
        Stream["Spectrum"] = SignalProcessingUtility.defaultSpectrogram(filtered,
                                                      window=1.0, overlap=0.5,
                                                      frequency_resolution=0.5,
                                                      fs=Stream["SamplingRate"])
        Timestamp.append(datetime.fromisoformat(Stream["FirstPacketDateTime"][:-1]+"+00:00").timestamp())
    Timestamp = np.array(Timestamp)

    #UniqueSessions = np.flip(sorted(np.unique(Timestamp)))
    UniqueSessions = list()
    for session in Timestamp:
        if np.sum(np.bitwise_and(np.array(UniqueSessions) > session - 60, np.array(UniqueSessions) < session + 60)) == 0:
            UniqueSessions.append(session)
    UniqueSessions = np.flip(sorted(UniqueSessions))

    PossibleSessions = list()
    PossibleSessions.append({"Index": 0, "Label": "Most Recent"})
    for sessionIndex in range(len(UniqueSessions)):
        PossibleSessions.append({"Index": sessionIndex, "Label": datetime.fromtimestamp(UniqueSessions[sessionIndex]).strftime("%Y %b %d, %I:%M %p")})

    uniqueTimestamp = UniqueSessions[index]

    # Visualization Sections of Baseline PSDs
    trialLegends = (list(),list())
    cmap = plt.get_cmap("Set1", 9)

    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=("Left Hemisphere", "Right Hemisphere"))

    for i in range(len(Data["SurveyPSD"])):
        if Timestamp[i] > uniqueTimestamp - 60 and Timestamp[i] < uniqueTimestamp + 60:
            (channels, hemisphere) = Percept.reformatChannelName(Data["SurveyPSD"][i]["Channel"])

            colorText = "rgb(0,0,0)"
            legendGroupName = f"E{channels[0]}-E{channels[1]}"
            if channels == [0,1]:
                colorText = "rgb(252,44,3)"
            elif channels == [0,2]:
                colorText = "rgb(252,161,3)"
            elif channels == [0,3]:
                colorText = "rgb(100,100,100)"
            elif channels == [1,2]:
                colorText = "rgb(3,252,53)"
            elif channels == [1,3]:
                colorText = "rgb(3,252,186)"
            elif channels == [2,3]:
                colorText = "rgb(3,132,252)"

            elif channels == [1.1,2.1]:
                colorText = "rgb(255,0,0)"
                legendGroupName = "Segmented Side 01"
            elif channels == [1.2,2.2]:
                colorText = "rgb(0,255,0)"
                legendGroupName = "Segmented Side 01"
            elif channels == [1.3,2.3]:
                colorText = "rgb(0,0,255)"
                legendGroupName = "Segmented Side 01"

            elif channels == [1.1,1.2]:
                colorText = "rgb(128,128,0)"
                legendGroupName = "Segmented Ring 01"
            elif channels == [1.1,1.3]:
                colorText = "rgb(128,0,128)"
                legendGroupName = "Segmented Ring 01"
            elif channels == [1.2,1.3]:
                colorText = "rgb(0,128,128)"
                legendGroupName = "Segmented Ring 01"

            elif channels == [2.1,2.2]:
                colorText = "rgb(228,228,0)"
                legendGroupName = "Segmented Ring 02"
            elif channels == [2.1,2.3]:
                colorText = "rgb(228,0,228)"
                legendGroupName = "Segmented Ring 02"
            elif channels == [2.2,2.3]:
                colorText = "rgb(0,228,228)"
                legendGroupName = "Segmented Ring 02"

            if hemisphere == "Left":
                SignalData = Data["SurveyPSD"][i]["Spectrum"]["Power"]
                """
                fig = addShadedErrorBar(fig, Data["SurveyPSD"][i]["Spectrum"]["Frequency"],
                                            np.mean(SignalData,axis=1),
                                            2*SignalProcessingUtility.stderr(SignalData, axis=1).flatten(),
                                            color=colorTextFromCmap(colorText),
                                            alpha=0.3,
                                            legendgroup=legendGroupName,
                                            row=1, col=1)
                """
                fig.add_trace(
                    go.Scatter(x=Data["SurveyPSD"][i]["Spectrum"]["Frequency"],
                               y=np.mean(SignalData,axis=1),
                               mode="lines",
                               name="E{0}-E{1}".format(channels[0],channels[1]),
                               line=dict(color=colorText, width=2),
                               legendgroup=legendGroupName,
                               hovertemplate="E{0}-E{1}".format(channels[0],channels[1]) + "  %{y:.2f} μV<sup>2</sup>/Hz <extra></extra>",
                               showlegend=True),
                    row=1, col=1
                )

            if hemisphere == "Right":
                SignalData = Data["SurveyPSD"][i]["Spectrum"]["Power"]
                """
                fig = addShadedErrorBar(fig, Data["SurveyPSD"][i]["Spectrum"]["Frequency"],
                                            np.mean(SignalData,axis=1),
                                            2*SignalProcessingUtility.stderr(SignalData, axis=1).flatten(),
                                            color=colorTextFromCmap(colorText),
                                            alpha=0.3,
                                            legendgroup=legendGroupName,
                                            row=1, col=2)
                """
                fig.add_trace(
                    go.Scatter(x=Data["SurveyPSD"][i]["Spectrum"]["Frequency"],
                               y=np.mean(SignalData,axis=1),
                               mode="lines",
                               name="E{0}-E{1}".format(channels[0],channels[1]),
                               line=dict(color=colorText, width=2),
                               legendgroup=legendGroupName,
                               hovertemplate="E{0}-E{1}".format(channels[0],channels[1]) + "  %{y:.2f} μV<sup>2</sup>/Hz <extra></extra>",
                               showlegend=True),

                    row=1, col=2
                )


    defaultYAxis = np.logspace(-3,2,num=6)
    fig.update_yaxes(type='log', range=(-3,2),
                     title_font_size=15, title_text="Power (μV<sup>2</sup>/Hz)", tickfont_size=12,
                     tickmode="array", tickvals=defaultYAxis, ticktext=[f"{i}" for i in defaultYAxis],
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     row=1, col=1)
    fig.update_yaxes(type='log', range=(-3,2),
                     title_font_size=15, title_text="Power (μV<sup>2</sup>/Hz)", tickfont_size=12,
                     tickmode="array", tickvals=defaultYAxis, ticktext=[f"{i}" for i in defaultYAxis],
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     row=1, col=2)
    fig.update_xaxes(type="linear", range=(0,100),
                     title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     row=1, col=1)
    fig.update_xaxes(type="linear", range=(0,100),
                     title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     row=1, col=2)
    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"
    fig.update_layout(hovermode="x")

    UniqueSessionPSDsDiv = po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div')

    return UniqueSessionPSDsDiv, PossibleSessions

def getSpectralChronicle(Data, channelName=None, grouping=False):
    if "SurveyPSD" in Data.keys():
        ChannalDescriptions = list()
        for i in range(len(Data["SurveyPSD"])):
            channels, hemisphere = Percept.reformatChannelName(Data["SurveyPSD"][i]["Channel"])
            ChannalDescriptions.append(f"{hemisphere}Hemisphere_E{channels[0]}_E{channels[1]}")
        UniqueChannels = sorted(PythonUtility.uniqueList(ChannalDescriptions))

        if len(UniqueChannels) == 0:
            return "", []

        if channelName == None:
            channelName = UniqueChannels[0]

        SpectralPowers = list()
        for i in range(len(ChannalDescriptions)):
            if ChannalDescriptions[i] == channelName:
                SpectralPowers.append(np.mean(Data["SurveyPSD"][i]["Spectrum"]["Power"], axis=1))
        SpectralPowers = np.array(SpectralPowers)

        if grouping:
            Timestamp = [datetime.fromisoformat(Data["SurveyPSD"][i]["FirstPacketDateTime"][:-1]+"+00:00").replace(hour=0, minute=0, second=0, microsecond=0).timestamp() for i in range(len(ChannalDescriptions)) if ChannalDescriptions[i] == channelName]
            UniqueTimestamp = sorted(np.unique(np.array(Timestamp)))
        else:
            Timestamp = [datetime.fromisoformat(Data["SurveyPSD"][i]["FirstPacketDateTime"][:-1]+"+00:00").timestamp() for i in range(len(ChannalDescriptions)) if ChannalDescriptions[i] == channelName]
            UniqueTimestamp = sorted(np.unique(np.array(Timestamp)))

        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=[f"{channelName} Baseline PSDs"])
        cmap = plt.get_cmap("jet", len(UniqueTimestamp)); cIndex = 0

        for i in range(len(UniqueTimestamp)):
            colorInfo = np.array(cmap(cIndex)[:-1]) * 255
            colorText = f"rgb({int(colorInfo[0])},{int(colorInfo[1])},{int(colorInfo[2])})"
            cIndex += 1

            if grouping:
                PSDName = datetime.fromtimestamp(UniqueTimestamp[i]).strftime("%Y %b %d")
            else:
                PSDName = datetime.fromtimestamp(UniqueTimestamp[i]).strftime("%Y %b %d - %H:%M:%S")

            SignalData = SpectralPowers[np.array(Timestamp) == UniqueTimestamp[i],:]
            fig.add_trace(
                go.Scatter(x=Data["SurveyPSD"][0]["Spectrum"]["Frequency"],
                           y=np.mean(SignalData,axis=0),
                           mode="lines",
                           name=PSDName,
                           line=dict(color=colorText, width=2),
                           hovertemplate=PSDName + "  %{y:.2f} μV<sup>2</sup>/Hz <extra></extra>",
                           showlegend=True),

                row=1, col=1
            )

        defaultYAxis = np.logspace(-3,2,num=6)
        fig.update_yaxes(type='log', range=(-3,2),
                         title_font_size=15, title_text="Power (μV<sup>2</sup>/Hz)", tickfont_size=12,
                         tickmode="array", tickvals=defaultYAxis, ticktext=[f"{i}" for i in defaultYAxis],
                         color="#000000", ticks="outside", showline=True, linecolor="#000000",
                         showgrid=True, gridcolor="#DDDDDD",
                         row=1, col=1)
        fig.update_xaxes(type="linear", range=(0,100),
                         title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                         color="#000000", ticks="outside", showline=True, linecolor="#000000",
                         showgrid=True, gridcolor="#DDDDDD",
                         row=1, col=1)
        fig.layout["paper_bgcolor"]="#FFFFFF"
        fig.layout["plot_bgcolor"]="#FFFFFF"
        fig.update_layout(hovermode="x")

        return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div'), UniqueChannels

def getMontageStream(Data, nItem = 0, reference=False):
    TotalChannels = len(Data["MontageStream"][nItem])
    StreamTitle = datetime.fromisoformat(Data["MontageStream"][nItem][0]["FirstPacketDateTime"][:-4]+"+00:00").astimezone(dateutil.tz.tzlocal()).strftime("%Y %B %d, %A, %I:%M:%S %p %Z")
    subplot_titles = list()
    for nStream in range(TotalChannels):
        (channels, hemisphere) = Percept.reformatChannelName(Data["MontageStream"][nItem][nStream]["Channel"])
        subplot_titles.append(f"{hemisphere} Hemisphere E{channels[0]}-E{channels[1]} Time-domain LFP")

    fig = make_subplots(rows=TotalChannels, cols=1, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing = 0.05,
                        subplot_titles=subplot_titles)

    [b,a] = signal.butter(5, np.array([1,25])*2/250, 'bp', output='ba')
    for nStream in range(TotalChannels):
        filtered = signal.filtfilt(b, a, Data["MontageStream"][nItem][nStream]["Data"])
        fig.add_trace(
                        go.Scatter(x=Data["MontageStream"][nItem][nStream]["Time"][0:-1:5],
                                   y=filtered[0:-1:5],
                                   mode="lines",
                                   line=dict(color="#000000", width=0.5),
                                   hovertemplate="  %{y:.2f} μV <extra></extra>",
                                   showlegend=False),
                        row=nStream+1, col=1
                    )

        fig.update_yaxes(type='linear', range=(-200,200),
                        title_font_size=15, title_text="Amplitude (μV)", tickfont_size=12,
                        color="#000000", ticks="outside", showline=True, linecolor="#000000",
                        showgrid=True, gridcolor="#DDDDDD",
                        row=nStream+1, col=1)

        fig.update_xaxes(color="#000000", ticks="outside", showline=True, linecolor="#000000",
                         showgrid=True, gridcolor="#DDDDDD",
                         row=nStream+1, col=1)


    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"

    fig2 = make_subplots(rows=TotalChannels, cols=1, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing = 0.05,
                        subplot_titles=subplot_titles)

    for nStream in range(TotalChannels):

        fig2.add_heatmap(x = Data["MontageStream"][nItem][nStream]["Spectrum"]["Time"],
                        y = Data["MontageStream"][nItem][nStream]["Spectrum"]["Frequency"],
                        z = np.log10(Data["MontageStream"][nItem][nStream]["Spectrum"]["Power"]),
                        coloraxis = "coloraxis", zsmooth="best",
                        hovertemplate="  %{x:.1f} sec<br>" +
                                      "  %{y:.1f} Hz<br>" +
                                      "  %{z:.2f} dB<extra></extra>",
                        row=nStream+1, col=1
                    )

        fig2.update_yaxes(type='linear', range=(0,100),
                        title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                        color="#000000", ticks="outside", showline=True, linecolor="#000000",
                        row=nStream+1, col=1)

        fig2.update_xaxes(color="#000000", ticks="outside", showline=True, linecolor="#000000",
                         showgrid=True, gridcolor="#DDDDDD",
                         row=nStream+1, col=1)

    if reference:
        fig2.update_coloraxes(colorbar_y=0.5, colorbar_len=1, showscale=True,
                             colorscale="Jet", cmin=-2, cmax=2,
                             colorbar_title_text = "Power (dB)", colorbar_title_side="right",
                             colorbar_title_font_size=15, colorbar_tickfont_size=12)
    else:
        fig2.update_coloraxes(colorbar_y=0.5, colorbar_len=1, showscale=True,
                             colorscale="Jet", cmin=-3, cmax=1,
                             colorbar_title_text = "Power (dB)", colorbar_title_side="right",
                             colorbar_title_font_size=15, colorbar_tickfont_size=12)

    fig2.layout["paper_bgcolor"]="#FFFFFF"
    fig2.layout["plot_bgcolor"]="#FFFFFF"

    return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div'), po.plot(fig2, auto_open=False, include_plotlyjs=False, output_type='div'), StreamTitle

def getTimeFrequencyAnalysis(Data, nStream = 0, normalized=False):
    (channels, hemisphere) = Percept.reformatChannelName(Data["StreamingTD"][nStream]["Channel"])
    if hemisphere == "Left":
        StimulationSide = 0
    else:
        StimulationSide = 1

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, shared_yaxes=False,
                        vertical_spacing = 0.05,
                        subplot_titles=(f"{hemisphere} Hemisphere E{channels[0]}-E{channels[1]} Time-domain LFP",
                                        "Time-Frequency Analysis",
                                        f"Power Domain ({Data['StreamingPower'][nStream]['TherapySnapshot'][hemisphere]['FrequencyInHertz']}Hz)",
                                        f"Stimulation State ({Data['StreamingPower'][nStream]['TherapySnapshot'][hemisphere]['RateInHertz']}Hz)"))

    [b,a] = signal.butter(5, np.array([1,25])*2/250, 'bp', output='ba')
    fig.add_trace(
                    go.Scatter(x=Data["StreamingTD"][nStream]["Time"][0:-1:5],
                               y=signal.filtfilt(b,a,Data["StreamingTD"][nStream]["Data"])[0:-1:5],
                               mode="lines",
                               line=dict(color="#000000", width=0.5),
                               hovertemplate="  %{y:.2f} μV <extra></extra>",
                               showlegend=False),
                    row=1, col=1
                )

    if normalized:
        Stimulation = np.interp(Data["StreamingTD"][nStream]["Spectrum"]["Time"], Data["StreamingPower"][nStream]["Time"], np.sum(Data["StreamingPower"][nStream]["Stimulation"],axis=1))
        LeastStimulationAmplitude = np.percentile(Stimulation,10)
        TimeSinceStimulationChange = np.interp(Data["StreamingTD"][nStream]["Spectrum"]["Time"], Data["StreamingPower"][nStream]["Time"][:len(Data["StreamingPower"][nStream]["TimeSinceStimulationChange"])], Data["StreamingPower"][nStream]["TimeSinceStimulationChange"][:,StimulationSide])
        BaselineSelection = np.bitwise_and(Stimulation <= LeastStimulationAmplitude,
                                            TimeSinceStimulationChange > 0)

        Power = np.log10(Data["StreamingTD"][nStream]["Spectrum"]["Power"])
        Baseline = np.mean(Power[:,BaselineSelection],axis=1)

        for i in range(Power.shape[1]):
            Power[:,i] -= Baseline

        fig.add_heatmap(x = Data["StreamingTD"][nStream]["Spectrum"]["Time"],
                        y = Data["StreamingTD"][nStream]["Spectrum"]["Frequency"],
                        z = Power,
                        coloraxis = "coloraxis", zsmooth="best",
                        hovertemplate="  %{x:.1f} sec<br>" +
                                      "  %{y:.1f} Hz<br>" +
                                      "  %{z:.2f} dB<extra></extra>",
                        row=2, col=1
                    )

    else:
        fig.add_heatmap(x = Data["StreamingTD"][nStream]["Spectrum"]["Time"],
                        y = Data["StreamingTD"][nStream]["Spectrum"]["Frequency"],
                        z = np.log10(Data["StreamingTD"][nStream]["Spectrum"]["Power"]),
                        coloraxis = "coloraxis", zsmooth="best",
                        hovertemplate="  %{x:.1f} sec<br>" +
                                      "  %{y:.1f} Hz<br>" +
                                      "  %{z:.2f} dB<extra></extra>",
                        row=2, col=1
                    )

    fig.add_trace(
                    go.Scatter(x=Data["StreamingPower"][nStream]["Time"],
                               y=Data["StreamingPower"][nStream]["Power"][:,StimulationSide],
                               mode="lines",
                               line=dict(color="#000000", width=3),
                               hovertemplate="  %{y:.2f} <extra></extra>",
                               showlegend=False),
                    row=3, col=1
                )

    hemisphere = "Left"
    if hemisphere in Data['StreamingPower'][nStream]['TherapySnapshot'].keys():
        Pulsewidth = f"{Data['StreamingPower'][nStream]['TherapySnapshot'][hemisphere]['PulseWidthInMicroSecond']} μs"
    else:
        Pulsewidth = "OFF"
    fig.add_trace(
                    go.Scatter(x=Data["StreamingPower"][nStream]["Time"],
                               y=Data["StreamingPower"][nStream]["Stimulation"][:,0],
                               mode="lines",
                               name=f"Left Stimulation ({Pulsewidth})",
                               line=dict(color="#FCA503", width=2),
                               hovertemplate=" Left %{y:.2f}mA<extra></extra>",
                               showlegend=True),
                    row=4, col=1
                )

    hemisphere = "Right"
    if hemisphere in Data['StreamingPower'][nStream]['TherapySnapshot'].keys():
        Pulsewidth = f"{Data['StreamingPower'][nStream]['TherapySnapshot'][hemisphere]['PulseWidthInMicroSecond']} μs"
    else:
        Pulsewidth = "OFF"
    fig.add_trace(
                    go.Scatter(x=Data["StreamingPower"][nStream]["Time"],
                               y=Data["StreamingPower"][nStream]["Stimulation"][:,1],
                               mode="lines",
                               name=f"Right Stimulation ({Pulsewidth})",
                               line=dict(color="#253EF7", width=2),
                               hovertemplate="Right %{y:.2f}mA<extra></extra>",
                               showlegend=True),
                    row=4, col=1
                )

    fig.update_yaxes(type='linear', range=(-200,200),
                    title_font_size=15, title_text="Amplitude (μV)", tickfont_size=12,
                    color="#000000", ticks="outside", showline=True, linecolor="#000000",
                    showgrid=True, gridcolor="#DDDDDD",
                    row=1, col=1)

    fig.update_xaxes(color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     row=1, col=1)

    if Data["StreamingTD"][nStream]["Spectrum"]["Type"] == "Wavelet":
        fig.update_yaxes(type='linear', range=(0,100),
                        title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                        color="#000000", ticks="outside", showline=True, linecolor="#000000",
                        row=2, col=1)
        """
        defaultYAxis = [1,4,12,30,60,100]
        fig.update_yaxes(type='log', range=(0,2),
                        title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                        tickmode="array", tickvals=defaultYAxis, ticktext=[f"{i}" for i in defaultYAxis],
                        color="#000000", ticks="outside", showline=True, linecolor="#000000",
                        row=2, col=1)
        """
    else:
        fig.update_yaxes(type='linear', range=(0,100),
                        title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                        color="#000000", ticks="outside", showline=True, linecolor="#000000",
                        row=2, col=1)

    fig.update_xaxes(color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     row=2, col=1)

    IQTR = np.percentile(Data["StreamingPower"][nStream]["Power"][:,StimulationSide], 85) - np.percentile(Data["StreamingPower"][nStream]["Power"][:,StimulationSide], 15)
    fig.update_yaxes(type='linear', range=(np.median(Data["StreamingPower"][nStream]["Power"][:,StimulationSide])-IQTR,np.median(Data["StreamingPower"][nStream]["Power"][:,StimulationSide])+IQTR),
                    title_font_size=15, title_text="Power (a.u.)", tickfont_size=12,
                    color="#000000", ticks="outside", showline=True, linecolor="#000000",
                    showgrid=True, gridcolor="#DDDDDD",
                    row=3, col=1)

    fig.update_xaxes(color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     row=3, col=1)

    fig.update_yaxes(type='linear', range=(0,5),
                    title_font_size=15, title_text="Stimulation (mA)", tickfont_size=12,
                    color="#000000", ticks="outside", showline=True, linecolor="#000000",
                    showgrid=True, gridcolor="#DDDDDD",
                    row=4, col=1)

    fig.update_xaxes(type="linear", range=(Data["StreamingTD"][nStream]["Spectrum"]["Time"][0], Data["StreamingTD"][nStream]["Spectrum"]["Time"][-1]),
                     title_font_size=15, title_text="Time (sec)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     row=4, col=1)

    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.10,
    ))

    if normalized:
        if Data["StreamingTD"][nStream]["Spectrum"]["Type"] == "Wavelet":
            fig.update_coloraxes(colorbar_y=0.63125, colorbar_len=0.25, showscale=True,
                                 colorscale="Jet", cmin=-1.5, cmax=1.5,
                                 colorbar_title_text = "Power (dB)", colorbar_title_side="right",
                                 colorbar_title_font_size=15, colorbar_tickfont_size=12)
        else:
            fig.update_coloraxes(colorbar_y=0.63125, colorbar_len=0.25, showscale=True,
                                 colorscale="Jet", cmin=-2, cmax=2,
                                 colorbar_title_text = "Power (dB)", colorbar_title_side="right",
                                 colorbar_title_font_size=15, colorbar_tickfont_size=12)
    else:
        if Data["StreamingTD"][nStream]["Spectrum"]["Type"] == "Wavelet":
            fig.update_coloraxes(colorbar_y=0.63125, colorbar_len=0.25, showscale=True,
                                 colorscale="Jet", cmin=0, cmax=2,
                                 colorbar_title_text = "Power (dB)", colorbar_title_side="right",
                                 colorbar_title_font_size=15, colorbar_tickfont_size=12)
        else:
            fig.update_coloraxes(colorbar_y=0.63125, colorbar_len=0.25, showscale=True,
                                 colorscale="Jet", cmin=-3, cmax=1,
                                 colorbar_title_text = "Power (dB)", colorbar_title_side="right",
                                 colorbar_title_font_size=15, colorbar_tickfont_size=12)

    return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div')

def getTherapyEffects(Data, nStream=0):
    if len(Data["StreamingTD"][nStream]["StimulationEpochs"]) == 0:
        return "", ""

    fig = go.Figure()
    cmap = plt.get_cmap("jet", len(Data["StreamingTD"][nStream]["StimulationEpochs"])); cIndex = 0

    for n in range(len(Data["StreamingTD"][nStream]["StimulationEpochs"])):
        colorInfo = np.array(cmap(cIndex)[:-1]) * 255
        colorText = f"rgb({int(colorInfo[0])},{int(colorInfo[1])},{int(colorInfo[2])})"
        cIndex += 1

        fig.add_trace(
                        go.Scatter(x=Data["StreamingTD"][nStream]["StimulationEpochs"][n]["Frequency"],
                                   y=Data["StreamingTD"][nStream]["StimulationEpochs"][n]["PSD"],
                                   mode="lines",
                                   name=f"{Data['StreamingTD'][nStream]['StimulationEpochs'][n]['Stimulation']} mA",
                                   line=dict(color=colorText, width=2),
                                   hovertemplate="%{meta} %{y:.2f} μV<sup>2</sup>/Hz<extra></extra>",
                                   meta=f"{Data['StreamingTD'][nStream]['StimulationEpochs'][n]['Stimulation']} mA",
                                   showlegend=True)
                    )

    defaultYAxis = np.logspace(-3,2,num=6)
    fig.update_yaxes(type='log', range=(-3,2),
                     title_font_size=15, title_text="Power (μV<sup>2</sup>/Hz)", tickfont_size=12,
                     tickmode="array", tickvals=defaultYAxis, ticktext=[f"{i}" for i in defaultYAxis],
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     zeroline=True, zerolinecolor="#DDDDDD",
                     showgrid=True, gridcolor="#DDDDDD")

    fig.update_xaxes(type="linear", range=(0, 100),
                     title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD")

    fig.update_layout(hovermode="x")
    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"
    fig.layout["height"] = 600

    fig2 = go.Figure()
    cmap = plt.get_cmap("jet", len(Data["StreamingTD"][nStream]["StimulationEpochs"])); cIndex = 0

    for n in range(len(Data["StreamingTD"][nStream]["StimulationEpochs"])):
        colorInfo = np.array(cmap(cIndex)[:-1]) * 255
        colorText = f"rgb({int(colorInfo[0])},{int(colorInfo[1])},{int(colorInfo[2])})"

        fig2.add_trace(
                        go.Scatter(x=Data["StreamingTD"][nStream]["StimulationEpochs"][n]["Frequency"],
                                   y=Data["StreamingTD"][nStream]["StimulationEpochs"][n]["NormalizedPSD"],
                                   mode="lines",
                                   name=f"{Data['StreamingTD'][nStream]['StimulationEpochs'][n]['Stimulation']} mA",
                                   line=dict(color=colorText, width=2),
                                   hovertemplate="%{meta} %{y:.2f} <extra></extra>",
                                   meta=f"{Data['StreamingTD'][nStream]['StimulationEpochs'][n]['Stimulation']} mA",
                                   showlegend=True)
                    )

        cIndex += 1

    defaultYAxis = np.logspace(-2,2,num=5)
    fig2.update_yaxes(type='log', range=(-2,2),
                     title_font_size=15, title_text="Normalized Power", tickfont_size=12,
                     tickmode="array", tickvals=defaultYAxis, ticktext=[f"{i}" for i in defaultYAxis],
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     zeroline=True, zerolinecolor="#DDDDDD",
                     showgrid=True, gridcolor="#DDDDDD")

    fig2.update_xaxes(type="linear", range=(0, 100),
                     title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD")

    fig2.update_layout(hovermode="x")
    fig2.layout["paper_bgcolor"]="#FFFFFF"
    fig2.layout["plot_bgcolor"]="#FFFFFF"
    fig2.layout["height"] = 600

    return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div'), po.plot(fig2, auto_open=False, include_plotlyjs=False, output_type='div')

def getTherapyEffects_SFFT(Data, nStream=0):
    if len(Data["StreamingTD"][nStream]["StimulationEpochs"]) == 0:
        return "", ""

    fig = go.Figure()
    cmap = plt.get_cmap("jet", len(Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"])); cIndex = 0

    for n in range(len(Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"])):
        if np.sum([Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][n]]) > 15:
            colorInfo = np.array(cmap(cIndex)[:-1]) * 255
            colorText = f"rgb({int(colorInfo[0])},{int(colorInfo[1])},{int(colorInfo[2])})"

            fig = addShadedErrorBar(fig, Data["StreamingTD"][nStream]["Spectrum"]["Frequency"],
                                        np.mean(Data["StreamingTD"][nStream]["Spectrum"]["SegmentedPower"][:, Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][n]], axis=1),
                                        2*SignalProcessingUtility.stderr(Data["StreamingTD"][nStream]["Spectrum"]["SegmentedPower"][:, Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][n]], axis=1).flatten(),
                                        color=colorTextFromCmap(cmap(cIndex)),
                                        alpha=0.3,
                                        legendgroup=f"{Data['StreamingTD'][nStream]['Spectrum']['StimulationLevels'][n]} mA")

            fig.add_trace(
                            go.Scatter(x=Data["StreamingTD"][nStream]["Spectrum"]["Frequency"],
                                       y=np.mean(Data["StreamingTD"][nStream]["Spectrum"]["SegmentedPower"][:, Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][n]], axis=1),
                                       mode="lines",
                                       name=f"{Data['StreamingTD'][nStream]['Spectrum']['StimulationLevels'][n]} mA",
                                       line=dict(color=colorText, width=2),
                                       hovertemplate="%{meta} %{y:.2f} μV<sup>2</sup>/Hz<extra></extra>",
                                       meta=f"{Data['StreamingTD'][nStream]['Spectrum']['StimulationLevels'][n]} mA",
                                       legendgroup=f"{Data['StreamingTD'][nStream]['Spectrum']['StimulationLevels'][n]} mA",
                                       showlegend=True)
                        )

        cIndex += 1

    defaultYAxis = np.logspace(-3,2,num=6)
    fig.update_yaxes(type='log', range=(-3,2),
                     title_font_size=15, title_text="Power (μV<sup>2</sup>/Hz)", tickfont_size=12,
                     tickmode="array", tickvals=defaultYAxis, ticktext=[f"{i}" for i in defaultYAxis],
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     zeroline=True, zerolinecolor="#DDDDDD",
                     showgrid=True, gridcolor="#DDDDDD")

    fig.update_xaxes(type="linear", range=(0, 100),
                     title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD")

    fig.update_layout(hovermode="x")
    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"
    fig.layout["height"] = 600

    fig2 = go.Figure()
    cmap = plt.get_cmap("jet", len(Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"])); cIndex = 0

    for n in range(len(Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"])):
        if np.sum([Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][n]]) > 15:
            colorInfo = np.array(cmap(cIndex)[:-1]) * 255
            colorText = f"rgb({int(colorInfo[0])},{int(colorInfo[1])},{int(colorInfo[2])})"

            fig2 = addShadedErrorBar(fig2, Data["StreamingTD"][nStream]["Spectrum"]["Frequency"],
                                        np.mean(Data["StreamingTD"][nStream]["Spectrum"]["NormalizedPower"][:, Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][n]], axis=1),
                                        2*SignalProcessingUtility.stderr(Data["StreamingTD"][nStream]["Spectrum"]["NormalizedPower"][:, Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][n]], axis=1).flatten(),
                                        color=colorTextFromCmap(cmap(cIndex)),
                                        alpha=0.3,
                                        legendgroup=f"{Data['StreamingTD'][nStream]['Spectrum']['StimulationLevels'][n]} mA")


            fig2.add_trace(
                            go.Scatter(x=Data["StreamingTD"][nStream]["Spectrum"]["Frequency"],
                                       y=np.mean(Data["StreamingTD"][nStream]["Spectrum"]["NormalizedPower"][:, Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][n]], axis=1),
                                       mode="lines",
                                       name=f"{Data['StreamingTD'][nStream]['Spectrum']['StimulationLevels'][n]} mA",
                                       line=dict(color=colorText, width=2),
                                       hovertemplate="%{meta} %{y:.2f} μV<sup>2</sup>/Hz<extra></extra>",
                                       meta=f"{Data['StreamingTD'][nStream]['Spectrum']['StimulationLevels'][n]} mA",
                                       showlegend=True)
                        )
        cIndex += 1

    defaultYAxis = np.logspace(-2,2,num=5)
    fig2.update_yaxes(type='log', range=(-2,2),
                     title_font_size=15, title_text="Normalized Power", tickfont_size=12,
                     tickmode="array", tickvals=defaultYAxis, ticktext=[f"{i}" for i in defaultYAxis],
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     zeroline=True, zerolinecolor="#DDDDDD",
                     showgrid=True, gridcolor="#DDDDDD")

    fig2.update_xaxes(type="linear", range=(0, 100),
                     title_font_size=15, title_text="Frequency (Hz)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD")

    fig2.update_layout(hovermode="x")
    fig2.layout["paper_bgcolor"]="#FFFFFF"
    fig2.layout["plot_bgcolor"]="#FFFFFF"
    fig2.layout["height"] = 600

    return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div'), po.plot(fig2, auto_open=False, include_plotlyjs=False, output_type='div')

def getPowerDomainBox(Data, nStream=0):
    StimulationLevels = list()
    PowerFeatures = list()
    for level in range(len(Data["StreamingTD"][nStream]["StimulationEpochs"])):
        if not Data["StreamingTD"][nStream]["StimulationEpochs"][level]["Stimulation"] in StimulationLevels:
            if len(Data["StreamingTD"][nStream]["StimulationEpochs"][level]["PowerDomain"]) > 2:
                StimulationLevels.append(Data["StreamingTD"][nStream]["StimulationEpochs"][level]["Stimulation"])
                PowerFeatures.append(Data["StreamingTD"][nStream]["StimulationEpochs"][level]["PowerDomain"])
        else:
            StimIndex = np.where(StimulationLevels == Data["StreamingTD"][nStream]["StimulationEpochs"][level]["Stimulation"])[0].flatten()
            np.append(PowerFeatures[StimIndex[0]],Data["StreamingTD"][nStream]["StimulationEpochs"][level]["PowerDomain"])

    fig = go.Figure()
    statisticData = list()
    for level in range(len(StimulationLevels)):
        if len(PowerFeatures[level]) > 2:
            fig.add_trace(
                        go.Box(
                            x=StimulationLevels[level]*np.ones(PowerFeatures[level].shape),
                            y=PowerFeatures[level],
                            width=0.3,
                            name=f"{StimulationLevels[level]} mA",
                            marker_color='rgb(7,40,89)',
                            line_color='rgb(7,40,89)'
                        )
                    )
            statisticData.append({"Y": PowerFeatures[level], "X": StimulationLevels[level]})

    if len(statisticData) == 0:
        return ""

    try:
        fig.update_yaxes(type='linear', range=(0,np.max([np.percentile(PowerFeatures[n],99) for n in range(len(PowerFeatures))]) + 100),
                         title_font_size=15, title_text="Power (a.u.)", tickfont_size=12,
                         color="#000000", ticks="outside", showline=True, linecolor="#000000",
                         zeroline=True, zerolinecolor="#DDDDDD",
                         showgrid=True, gridcolor="#DDDDDD")
    except IndexError:
        fig.update_yaxes(type='linear', range=(0,3000),
                         title_font_size=15, title_text="Power (a.u.)", tickfont_size=12,
                         color="#000000", ticks="outside", showline=True, linecolor="#000000",
                         zeroline=True, zerolinecolor="#DDDDDD",
                         showgrid=True, gridcolor="#DDDDDD")

    fig.update_xaxes(type="linear", range=(-0.3, np.max(StimulationLevels)+0.3),
                     title_font_size=15, title_text="Stimulation Amplitude (mA)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD")

    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"
    fig.layout["height"] = 600

    (channels, hemisphere) = Percept.reformatChannelName(Data["StreamingTD"][nStream]["Channel"])
    senseFreq = Data["StreamingPower"][nStream]["TherapySnapshot"][hemisphere]["FrequencyInHertz"]
    fig.layout["title"] = {"text": f"{hemisphere} Hemisphere E{channels[0]}-E{channels[1]} {senseFreq}Hz Power",
                           "x": 0.5, "xanchor": "center"}

    return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div')

def getStatisticMap(Data, nStream=0, CenterFrequency=-1, Normalized=False):
    if not "PredictedCenterFrequency" in Data["StreamingTD"][nStream]["Spectrum"].keys():
        return ""

    if CenterFrequency < 0:
        CenterFrequency = Data["StreamingTD"][nStream]["Spectrum"]["PredictedCenterFrequency"]

    PowerSelection = np.bitwise_and(Data["StreamingTD"][nStream]["Spectrum"]["Frequency"] > CenterFrequency - 3, Data["StreamingTD"][nStream]["Spectrum"]["Frequency"] <= CenterFrequency + 3)

    if Normalized:
        SpectralFeature = np.mean(Data["StreamingTD"][nStream]["Spectrum"]["NormalizedPower"][PowerSelection,:],axis=0)
        fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=False,
                            horizontal_spacing = 0.15,
                            subplot_titles=(f"Normalized Power Value @ {CenterFrequency}Hz",
                                            "Statistic Map"))
    else:
        SpectralFeature = np.mean(Data["StreamingTD"][nStream]["Spectrum"]["SegmentedPower"][PowerSelection,:],axis=0)
        fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=False,
                            horizontal_spacing = 0.15,
                            subplot_titles=(f"Power Value @ {CenterFrequency}Hz",
                                            "Statistic Map"))

    statisticData = list()
    xdata = np.zeros(0)
    ydata = np.zeros(0)
    for level in range(len(Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"])):
        if len(SpectralFeature[Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level]]) > 15:
            fig.add_trace(
                        go.Box(
                            x=Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level]*np.ones(SpectralFeature[Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level]].shape),
                            y=SpectralFeature[Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level]],
                            width=0.2,
                            name=f"{Data['StreamingTD'][nStream]['Spectrum']['StimulationLevels'][level]} mA",
                            marker_color='rgb(46, 20, 105)',
                            line_color='rgb(46, 20, 105)',
                            hovertemplate="<extra></extra>",
                            showlegend=False
                        )
                    )
            FeaturePower = SpectralFeature[Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level]]
            FeaturePower = SignalProcessingUtility.removeOutlier(FeaturePower, method="zscore")
            statisticData.append({"Y": FeaturePower, "X": Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level]})
            #FeaturePower = SPU.removeOutlier(SPU.smooth(FeaturePower,5)[::5], method="zscore")
            #FeaturePower = np.mean(FeaturePower)
            ydata = np.append(ydata, FeaturePower)
            xdata = np.append(xdata, np.ones(FeaturePower.shape)*Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level])

    if len(statisticData) == 0:
        return ""

    PowerDecayFitFound = False
    try:
        scale = np.percentile(ydata,95)
        #ModelBounds = (0,[np.inf,20,np.inf,np.inf])
        ModelBounds = ([-np.inf,np.inf])
        parameters, fit_covariance = optimize.curve_fit(SignalProcessingUtility.PowerDecayFunc, xdata, ydata/scale, bounds=ModelBounds, method="trf")
        ConfidenceError = np.sqrt(np.diag(fit_covariance))
        PowerDecayError = np.mean(np.power(ydata/scale - SignalProcessingUtility.PowerDecayFunc(xdata, *parameters),2))
        fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
        fitted_line = SignalProcessingUtility.PowerDecayFunc(fitted_xdata, *parameters)*scale
        FitParameters = f"Power Decay Fitting Error: {PowerDecayError:.3}<br>Maximum Change: {parameters[0]:.3} ± {ConfidenceError[0]:.3}<br>Speed of Change: {parameters[1]:.3} ± {ConfidenceError[1]:.3}<br>Point of Change: {parameters[2]:.3} ± {ConfidenceError[2]:.3}<br>Minimum Power: {parameters[3]:.3} ± {ConfidenceError[3]:.3}"
        PowerDecayFitFound = True
    except:
        pass

    InverseSigmoidFound = False
    try:
        scale = np.percentile(ydata,95)
        #ModelBounds = (0,[np.inf,100,np.max(xdata),np.inf])
        ModelBounds = ([-np.inf,np.inf])
        parameters, fit_covariance = optimize.curve_fit(SignalProcessingUtility.InverseSigmoidFunc, xdata, ydata/scale, bounds=ModelBounds, method="trf")
        ConfidenceError = np.sqrt(np.diag(fit_covariance))
        ModelError = np.mean(np.power(ydata/scale - SignalProcessingUtility.InverseSigmoidFunc(xdata, *parameters),2))
        if PowerDecayFitFound:
            if ModelError <= PowerDecayError:
                fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
                fitted_line = SignalProcessingUtility.InverseSigmoidFunc(fitted_xdata, *parameters)*scale
                FitParameters = f"Inverse Sigmoid Fitting Error: {ModelError:.3}<br>Maximum Change: {parameters[0]:.3} ± {ConfidenceError[0]:.3}<br>Speed of Change: {parameters[1]:.3} ± {ConfidenceError[1]:.3}<br>Point of Change: {parameters[2]:.3} ± {ConfidenceError[2]:.3}<br>Minimum Power: {parameters[3]:.3} ± {ConfidenceError[3]:.3}"
                InverseSigmoidFound = True
        else:
            fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
            fitted_line = SignalProcessingUtility.InverseSigmoidFunc(fitted_xdata, *parameters)*scale
            FitParameters = f"Inverse Sigmoid Fitting Error: {ModelError:.3}<br>Maximum Change: {parameters[0]:.3} ± {ConfidenceError[0]:.3}<br>Speed of Change: {parameters[1]:.3} ± {ConfidenceError[1]:.3}<br>Point of Change: {parameters[2]:.3} ± {ConfidenceError[2]:.3}<br>Minimum Power: {parameters[3]:.3} ± {ConfidenceError[3]:.3}"
            InverseSigmoidFound = True
    except:
        pass


    if InverseSigmoidFound or PowerDecayFitFound:
        fig.add_trace(
                    go.Scatter(x=fitted_xdata,
                               y=fitted_line,
                               mode="lines",
                               name=FitParameters,
                               line=dict(color="rgb(100,0,0)", width=2),
                               hovertemplate=f"<extra></extra>",
                               showlegend=True)
                )

    try:
        fig.update_yaxes(type='linear', range=(0,np.max([np.percentile(statisticData[n]["Y"],99) for n in range(len(statisticData))]) + 0.1),
                         title_font_size=15, title_text="Power (a.u.)", tickfont_size=12,
                         color="#000000", ticks="outside", showline=True, linecolor="#000000",
                         zeroline=True, zerolinecolor="#DDDDDD",
                         showgrid=True, gridcolor="#DDDDDD")
    except IndexError:
        fig.update_yaxes(type='linear', range=(0,5),
                         title_font_size=15, title_text="Power (a.u.)", tickfont_size=12,
                         color="#000000", ticks="outside", showline=True, linecolor="#000000",
                         zeroline=True, zerolinecolor="#DDDDDD",
                         showgrid=True, gridcolor="#DDDDDD")


    fig.update_layout(legend=dict(
        xanchor="left",
        x=0.25,
        yanchor="top",
        y=0.95,
    ))

    totalTests = sum(range(len(statisticData)))
    matrixData = pd.DataFrame(dtype=float, columns=list(["Stimulation1","Stimulation2","StatisticMatrix","StatisticMatrixDirection"]))
    for index1 in range(len(statisticData)):
        for index2 in range(index1, len(statisticData)):
            pvalue = stats.ttest_ind(statisticData[index1]["Y"],statisticData[index2]["Y"],equal_var=False).pvalue
            pvalue = totalTests * pvalue
            if pvalue > 1:
                pvalue = 1
            matrixData = matrixData.append({"Stimulation1": statisticData[index1]["X"],
                              "Stimulation2": statisticData[index2]["X"],
                              "StatisticMatrix": np.abs(statisticData[index1]["Y"].mean() - statisticData[index2]["Y"].mean()),
                              "StatisticMatrixDirection": (1-pvalue) * np.sign(statisticData[index1]["Y"].mean() - statisticData[index2]["Y"].mean())}, ignore_index=True)

    cmap = plt.get_cmap("coolwarm",7)
    boundary = np.array([-1,-0.999,-0.99,-0.95,0.95,0.99,0.999,1]) / 2 + 0.5
    discreteColorScale = list()
    for i in range(7):
        colorInfo = np.array(cmap(i)[:-1]) * 255
        colorText = f"rgb({int(colorInfo[0])},{int(colorInfo[1])},{int(colorInfo[2])})"
        discreteColorScale.append([boundary[i], colorText])
        discreteColorScale.append([boundary[i+1], colorText])

    fig.add_trace(
                    go.Scatter(x=matrixData["Stimulation2"],
                               y=matrixData["Stimulation1"],
                               mode="markers",
                               marker= dict(
                                    size=matrixData["StatisticMatrix"]*8+5,
                                    color=matrixData["StatisticMatrixDirection"],
                                    colorscale=discreteColorScale,
                                    showscale=False,
                                    cmax=1, cmin=-1,
                                ),
                                hovertemplate= "p: %{text} <extra></extra>",
                                text=["{:.2}".format(i) for i in matrixData["StatisticMatrixDirection"]],
                                showlegend=False),
                    row=1, col=2
                )

    fig.update_yaxes(type="linear", range=(-0.3, np.max(Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"])+0.3),
                     title_font_size=15, title_text="Stimulation Amplitude (mA)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     zeroline=True, zerolinecolor="#DDDDDD",
                     showgrid=True, gridcolor="#DDDDDD", row=1, col=2)

    fig.update_xaxes(type="linear", range=(-0.3, np.max(Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"])+0.3),
                     title_font_size=15, title_text="Stimulation Amplitude (mA)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     zeroline=True, zerolinecolor="#DDDDDD",
                     showgrid=True, gridcolor="#DDDDDD")

    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"
    fig.layout["height"] = 600

    return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div')

def getTherapyHistory(Data):
    x = list()
    y = list()
    for changeHistory in Data["TherapyChangeHistory"]:
        x.append(changeHistory["DateTime"])
        y.append(changeHistory["OldGroup"])
        x.append(changeHistory["DateTime"])
        y.append(changeHistory["NewGroup"])
    x.append(Data["TherapyHistoryEndDate"])
    y.append(Data["TherapyChangeHistory"][-1]["NewGroup"])

    fig = go.Figure()
    fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       mode="lines",
                       line=dict(color="#FF0000", width=2),
                       hovertemplate="Date: %{x|%Y %B %d, %I:%M %p} <extra></extra>",
                       )
            )

    fig.update_yaxes(type="linear", range=(-0.5,3.5),
                     tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     zeroline=True, zerolinecolor="#DDDDDD",
                     tickvals=np.array(range(4)), ticktext=["Group A","Group B","Group C","Group D"])

    fig.update_xaxes(range=(Data["TherapyHistoryStartDate"],Data["TherapyHistoryEndDate"]),
                     tickformat="%Y %b %d\n%I:%M %p")

    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"

    return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div')

def getLFPTrends(Data):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.05,
                        subplot_titles=("Left Hemisphere LFP Trends",
                                        "Right Hemisphere LFP Trends",
                                        "Active Group"))

    firstTimePoint = datetime.utcnow().astimezone(dateutil.tz.tzlocal())
    lastTimePoint = datetime.fromtimestamp(0).astimezone(dateutil.tz.tzlocal())
    for PowerRecord in Data["PowerRecords"]:
        for hemisphere in PowerRecord["LFP"].keys():
            if hemisphere == "HemisphereLocationDef.Left":
                try:
                    SensingFrequency = PowerRecord["Therapy"]["LeftHemisphere"]["SensingSetup"]["FrequencyInHertz"]
                except:
                    SensingFrequency = -1

                try:
                    fig.add_trace(
                        go.Scattergl(x=PowerRecord["Time"][hemisphere],
                                   y=PowerRecord["LFP"][hemisphere].flatten(),
                                   mode="lines",
                                   line=dict(color="#000000", width=2),
                                   hovertemplate=Percept.reformatStimulationChannel(PowerRecord["Therapy"]["LeftHemisphere"]["Channel"]) + " " + str(SensingFrequency) + "Hz<br>Power: %{y}<br>Date: %{x|%Y %B %d, %I:%M %p} <extra></extra>",
                                   showlegend=False),
                            row = 1, col = 1
                        )
                    if PowerRecord["Time"][hemisphere][0] < firstTimePoint:
                        firstTimePoint = PowerRecord["Time"][hemisphere][0]
                    if PowerRecord["Time"][hemisphere][-1] > lastTimePoint:
                        lastTimePoint = PowerRecord["Time"][hemisphere][-1]
                except:
                    pass

            if hemisphere == "HemisphereLocationDef.Right":
                try:
                    SensingFrequency = PowerRecord["Therapy"]["RightHemisphere"]["SensingSetup"]["FrequencyInHertz"]
                except:
                    SensingFrequency = -1

                try:
                    fig.add_trace(
                        go.Scattergl(x=PowerRecord["Time"][hemisphere],
                                   y=PowerRecord["LFP"][hemisphere].flatten(),
                                   mode="lines",
                                   line=dict(color="#000000", width=2),
                                   hovertemplate=Percept.reformatStimulationChannel(PowerRecord["Therapy"]["RightHemisphere"]["Channel"]) + " " + str(SensingFrequency) + "Hz<br>Power: %{y}<br>Date: %{x|%Y %B %d, %I:%M %p} <extra></extra>",
                                   showlegend=False),
                            row = 2, col = 1
                        )
                    if PowerRecord["Time"][hemisphere][0] < firstTimePoint:
                        firstTimePoint = PowerRecord["Time"][hemisphere][0]
                    if PowerRecord["Time"][hemisphere][-1] > lastTimePoint:
                        lastTimePoint = PowerRecord["Time"][hemisphere][-1]
                except:
                    pass


    fig.update_yaxes(type="linear", title="Power (a.u.)",
                     tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     zeroline=True, zerolinecolor="#DDDDDD",
              row = 1, col = 1)

    fig.update_yaxes(type="linear", title="Power (a.u.)",
                     tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     zeroline=True, zerolinecolor="#DDDDDD",
              row = 2, col = 1)

    x = list()
    y = list()
    for changeHistory in Data["TherapyChangeHistory"]:
        x.append(changeHistory["DateTime"])
        y.append(changeHistory["OldGroup"])
        x.append(changeHistory["DateTime"])
        y.append(changeHistory["NewGroup"])
    x.append(Data["TherapyHistoryEndDate"])
    y.append(Data["TherapyChangeHistory"][-1]["NewGroup"])

    fig.add_trace(
        go.Scattergl(x=x,
                   y=y,
                   mode="lines",
                   line=dict(color="#FF0000", width=2),
                   hovertemplate="Date: %{x|%Y %B %d, %I:%M %p} <extra></extra>",
                   showlegend=False
                   ),
            row = 3, col = 1
        )

    if "Events" in Data.keys():
        legendExist = list()
        allEvents = [event["EventName"] for event in Data["Events"]]
        UniqueEventName = PythonUtility.uniqueList(allEvents)
        cmap = plt.get_cmap("Set2", 9)

        for i in range(len(Data["Events"])):
            fig.add_trace(
                go.Scatter(x=[Data["Events"][i]["Time"],Data["Events"][i]["Time"]],
                           y=[-1,5],
                           mode="lines",
                           name=Data["Events"][i]["EventName"],
                           line=dict(color=colorTextFromCmap(cmap(np.where(PythonUtility.iterativeCompare(UniqueEventName,allEvents[i],"equal"))[0][0])), width=2),
                           hovertemplate="Date: %{x|%Y %B %d, %I:%M %p} <extra></extra>",
                           showlegend=not Data["Events"][i]["EventName"] in legendExist,
                           legendgroup=Data["Events"][i]["EventName"]
                           ),
                    row = 3, col = 1
                )
            legendExist.append(Data["Events"][i]["EventName"])

            fig.update_layout(legend=dict(
                yanchor="top",
                y=0.20,
            ))

    """
    if "PatientEventLogs" in Data.keys():
        legendExist = list()
        cmap = plt.get_cmap("Set2", 9)
        for i in range(len(Data["PatientEventLogs"])):
            fig.add_trace(
                go.Scatter(x=[Data["PatientEventLogs"][i]["DateTime"],Data["PatientEventLogs"][i]["DateTime"]],
                           y=[-1,5],
                           mode="lines",
                           name=Data["PatientEventLogs"][i]["EventName"],
                           line=dict(color=colorTextFromCmap(cmap(Data["PatientEventLogs"][i]["EventID"])), width=2),
                           hovertemplate="Date: %{x|%Y %B %d, %I:%M %p} <extra></extra>",
                           showlegend=not Data["PatientEventLogs"][i]["EventName"] in legendExist,
                           legendgroup=Data["PatientEventLogs"][i]["EventName"]
                           ),
                    row = 3, col = 1
                )
            legendExist.append(Data["PatientEventLogs"][i]["EventName"])

            fig.update_layout(legend=dict(
                yanchor="top",
                y=0.20,
            ))
    """
    fig.update_yaxes(type="linear", range=(-0.5,3.5),
                     tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     zeroline=True, zerolinecolor="#DDDDDD",
                     tickvals=np.array(range(4)), ticktext=["Group A","Group B","Group C","Group D"],
              row = 3, col = 1)

    fig.update_xaxes(range=(firstTimePoint,lastTimePoint), tickformat="%I:%M %p\n%b %d %Y")

    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"
    fig.layout["height"] = 900

    return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div')

def getAveragePSDGroup(Data, nGroup=0):

    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.05,
                    subplot_titles=("Left Hemisphere LFP Trends",
                                    "Right Hemisphere LFP Trends",
                                    "Active Group"))

    PowerRecord = Data["PowerRecords"][nGroup]
    PowerRecord["NormalizedTime"] = dict()
    for hemisphere in PowerRecord["LFP"].keys():
        if hemisphere == "HemisphereLocationDef.Left":
            try:
                SensingFrequency = PowerRecord["Therapy"]["LeftHemisphere"]["SensingSetup"]["FrequencyInHertz"]
            except:
                SensingFrequency = -1

            Title = Percept.reformatStimulationChannel(PowerRecord["Therapy"]["LeftHemisphere"]["Channel"]) + str(SensingFrequency) + "Hz"
            PowerRecord["NormalizedTime"][hemisphere] = np.array([PowerRecord]["Time"][hemisphere][t].replace(year=2000,month=1,day=1) for t in range(len(PowerRecord["Time"][hemisphere])))
            CircadianHours = np.array(range(0,24,1))
            Circadian = {"Time": list(), "Mean": list(), "Variance": list()}
            for h in range(len(CircadianHours)):
                SelectedData = PythonUtility.rangeSelection(PowerRecord["NormalizedTime"][hemisphere] - datetime(2000,1,1,tzinfo=dateutil.tz.tzlocal()), [timedelta(seconds=int(3600*CircadianHours[h])),timedelta(seconds=int(3600*CircadianHours[h]+3600))])
                Circadian["Mean"].append(np.mean(PowerRecord["LFP"][hemisphere][SelectedData]))
                Circadian["Variance"].append(np.std(PowerRecord["LFP"][hemisphere][SelectedData]))
                Circadian["Time"].append(datetime(year=2000,month=1,day=1,hour=CircadianHours[h],tzinfo=dateutil.tz.tzlocal()))
            Circadian["Time"] = np.array(Circadian["Time"])
            Circadian["Mean"] = np.array(Circadian["Mean"])
            Circadian["Variance"] = np.array(Circadian["Variance"])

        if hemisphere == "HemisphereLocationDef.Right":
            try:
                SensingFrequency = PowerRecord["Therapy"]["RightHemisphere"]["SensingSetup"]["FrequencyInHertz"]
            except:
                SensingFrequency = -1
            fig.add_trace(
                go.Scattergl(x=PowerRecord["Time"][hemisphere],
                           y=PowerRecord["LFP"][hemisphere].flatten(),
                           mode="lines",
                           line=dict(color="#000000", width=2),
                           hovertemplate=Percept.reformatStimulationChannel(PowerRecord["Therapy"]["RightHemisphere"]["Channel"]) + " " + str(SensingFrequency) + "Hz<br>Power: %{y}<br>Date: %{x|%Y %B %d, %I:%M %p} <extra></extra>",
                           showlegend=False),
                    row = 2, col = 1
                )
            if PowerRecord["Time"][hemisphere][0] < firstTimePoint:
                firstTimePoint = PowerRecord["Time"][hemisphere][0]
            if PowerRecord["Time"][hemisphere][-1] > lastTimePoint:
                lastTimePoint = PowerRecord["Time"][hemisphere][-1]

def getEventLockedPower(EventlockPowerData, therapyName=""):
    Therapies = [EventlockPowerData[i]["Configuration"] for i in range(len(EventlockPowerData))]
    UniqueTherapy = PythonUtility.uniqueList(Therapies)

    EventLockedPower = list()
    for i in range(len(UniqueTherapy)):
        try:
            SensingFrequency = UniqueTherapy[i]['Therapy']["SensingSetup"]["FrequencyInHertz"]
        except:
            SensingFrequency = -1

        Title = Percept.reformatStimulationChannel(UniqueTherapy[i]["Therapy"]["Channel"]) + " " + str(SensingFrequency) + "Hz"
        Hemisphere = UniqueTherapy[i]["Hemisphere"].replace("HemisphereLocationDef.","")
        Title = f"{Hemisphere} {UniqueTherapy[i]['Therapy']['Frequency']} Hz Stim " + Title

        if (therapyName == "" and i == 0) or (Title == therapyName):
            SelectedTherapy = np.where(PythonUtility.iterativeCompare(Therapies, UniqueTherapy[i], "equal").flatten())[0]
            PowerArray = np.zeros((len(EventlockPowerData[0]["Power"]),len(SelectedTherapy)))
            EventNames = [EventlockPowerData[i]["Event"] for i in SelectedTherapy]
            UniqueEvent = PythonUtility.uniqueList(EventNames)
            for j in range(len(SelectedTherapy)):
                PowerArray[:,j] = EventlockPowerData[j]["Power"]

            TimeRange = (np.arange(PowerArray.shape[0]) - PowerArray.shape[0] / 2) * 10

            fig = go.Figure()
            cmap = plt.get_cmap("Set1", 9)
            for j in range(len(UniqueEvent)):
                AveragePower = np.zeros(TimeRange.shape)
                StdPower = np.zeros(TimeRange.shape)
                for t in range(len(TimeRange)):
                    SelectedRow = np.bitwise_and(PowerArray[t,:].flatten()!=0, PythonUtility.iterativeCompare(EventNames, UniqueEvent[j], "equal").flatten())
                    AveragePower[t] = np.mean(PowerArray[t,SelectedRow])
                    StdPower[t] = np.std(PowerArray[t,SelectedRow])/np.sqrt(np.sum(SelectedRow))

                fig = addShadedErrorBar(fig, TimeRange, AveragePower, StdPower,
                                        color=colorTextFromCmap(cmap(j)),
                                        alpha=0.3,
                                        legendgroup=UniqueEvent[j] + f" (n={np.sum(PythonUtility.iterativeCompare(EventNames, UniqueEvent[j], 'equal'))})")
                fig.add_trace(
                        go.Scatter(x=TimeRange, y=AveragePower, mode="lines",
                                   line=dict(color=colorTextFromCmap(cmap(j)), width=2),
                                   name=UniqueEvent[j] + f" (n={np.sum(PythonUtility.iterativeCompare(EventNames, UniqueEvent[j], 'equal'))})",
                                   legendgroup=UniqueEvent[j] + f" (n={np.sum(PythonUtility.iterativeCompare(EventNames, UniqueEvent[j], 'equal'))})",
                                   hovertemplate=UniqueEvent[j] + " Power: %{y} <extra></extra>",
                                   showlegend=True,
                                   )
                        )

            fig.update_yaxes(type="linear", range=(0, 3500),
                             title_font_size=15, title_text="Power (a.u.)", tickfont_size=12,
                             color="#000000", ticks="outside", showline=True, linecolor="#000000",
                             showgrid=True, gridcolor="#DDDDDD",
                             zeroline=True, zerolinecolor="#DDDDDD")

            fig.update_xaxes(type="linear", range=(TimeRange[0],TimeRange[-1]), title_text="Time (minutes)",
                             color="#000000", ticks="outside", showline=True, linecolor="#000000",
                             showgrid=True, gridcolor="#DDDDDD",
                             zeroline=True, zerolinecolor="#DDDDDD")

            fig.layout["paper_bgcolor"]="#FFFFFF"
            fig.layout["plot_bgcolor"]="#FFFFFF"
            fig.update_layout(hovermode="x")
            fig.update_layout(legend=dict( yanchor="top", y=0.99, xanchor="left", x=0.7 ),
                                title=dict( text=Title, x=0.5, font=dict(size=18)))

            EventLockedPower.append({"plotlydiv": po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div'), "therapyName": Title})
        else:
            EventLockedPower.append({"plotlydiv": "", "therapyName": Title})

    return EventLockedPower

def getUniqueConfiguration(Configurations):
    UniqueConfigurations = list()
    for rowID in range(len(Configurations)):
        unique = True
        for uniqueItem in UniqueConfigurations:
            if np.all(Configurations.loc[rowID,:] == uniqueItem):
                unique = False

        if unique:
            UniqueConfigurations.append(Configurations.loc[rowID,:])
    return UniqueConfigurations

def getAverageLFPperGroup(Data, ConfigurationText):
    Time = np.zeros(0)
    Power = np.zeros(0)
    for n in range(len(Data["PowerRecords"])):
        for hemisphere in Data["PowerRecords"][n]["LFP"].keys():
            if hemisphere == "HemisphereLocationDef.Left":
                if "LeftHemisphere" in Data["PowerRecords"][n]["Therapy"].keys():
                    if "SensingSetup" in Data["PowerRecords"][n]["Therapy"]["LeftHemisphere"].keys():
                        Configuration = dict()
                        ChannelName = Percept.reformatStimulationChannel(Data["PowerRecords"][n]["Therapy"]["LeftHemisphere"]["Channel"])
                        Configuration["Hemisphere"] = hemisphere
                        Configuration["Channel"] = ChannelName
                        Configuration["SensingFrequency"] = Data["PowerRecords"][n]["Therapy"]["LeftHemisphere"]["SensingSetup"]["FrequencyInHertz"]
                        Configuration["StimFrequency"] = Data["PowerRecords"][n]["Therapy"]["LeftHemisphere"]["Frequency"]
                        Configuration["StimPulseWidth"] = Data["PowerRecords"][n]["Therapy"]["LeftHemisphere"]["PulseWidth"]
                        CurrentText = f"{Configuration['Hemisphere'].replace('HemisphereLocationDef.','')} {Configuration['Channel']} - {Configuration['SensingFrequency']}Hz - Therapy @ {Configuration['StimFrequency']}Hz/{Configuration['StimPulseWidth']}us"

                        if ConfigurationText == CurrentText:
                            Time = np.append(Time, Data["PowerRecords"][n]["Time"]["HemisphereLocationDef.Left"])
                            Power = np.append(Power, Data["PowerRecords"][n]["LFP"]["HemisphereLocationDef.Left"])

            if hemisphere == "HemisphereLocationDef.Right":
                if "RightHemisphere" in Data["PowerRecords"][n]["Therapy"].keys():
                    if "SensingSetup" in Data["PowerRecords"][n]["Therapy"]["RightHemisphere"].keys():
                        Configuration = dict()
                        ChannelName = Percept.reformatStimulationChannel(Data["PowerRecords"][n]["Therapy"]["RightHemisphere"]["Channel"])
                        Configuration["Hemisphere"] = hemisphere
                        Configuration["Channel"] = ChannelName
                        Configuration["SensingFrequency"] = Data["PowerRecords"][n]["Therapy"]["RightHemisphere"]["SensingSetup"]["FrequencyInHertz"]
                        Configuration["StimFrequency"] = Data["PowerRecords"][n]["Therapy"]["RightHemisphere"]["Frequency"]
                        Configuration["StimPulseWidth"] = Data["PowerRecords"][n]["Therapy"]["RightHemisphere"]["PulseWidth"]
                        CurrentText = f"{Configuration['Hemisphere'].replace('HemisphereLocationDef.','')} {Configuration['Channel']} - {Configuration['SensingFrequency']}Hz - Therapy @ {Configuration['StimFrequency']}Hz/{Configuration['StimPulseWidth']}us"

                        if ConfigurationText == CurrentText:
                            Time = np.append(Time, Data["PowerRecords"][n]["Time"]["HemisphereLocationDef.Right"])
                            Power = np.append(Power, Data["PowerRecords"][n]["LFP"]["HemisphereLocationDef.Right"])

    if len(Time) == 0:
        return ""

    Time, index = np.unique(Time, return_index=True)
    Time = np.array([Time[i].timestamp() for i in range(len(Time))])
    LFPData = Power[index]

    CircadianClock = np.remainder(Time - datetime(year=2000,month=1,day=1,hour=4).timestamp(), 3600*24)

    HourlyIndex = np.arange(0, 3600*24, 600)
    AveragePower = np.zeros(HourlyIndex.shape)
    StdPower = np.zeros(HourlyIndex.shape)
    for i in range(len(HourlyIndex)):
        AveragePower[i] = np.mean(LFPData[PythonUtility.rangeSelection(CircadianClock,[HourlyIndex[i],HourlyIndex[i]+1800])])
        StdPower[i] = SignalProcessingUtility.stderr(LFPData[PythonUtility.rangeSelection(CircadianClock,[HourlyIndex[i],HourlyIndex[i]+1800])])

    fig = go.Figure()
    fig.add_trace(
            go.Scatter(x=CircadianClock / 3600,
                       y=LFPData,
                       mode="markers",
                       opacity=0.5,
                       hovertemplate="<extra></extra>",
                       showlegend=False,
                       )
            )

    fig = addShadedErrorBar(fig, HourlyIndex/3600, AveragePower.flatten(), StdPower.flatten(), color="#000000", alpha=0.4, legendgroup="averageLFPTrend")
    fig.add_trace(
        go.Scatter(x=HourlyIndex / 3600,
                   y=AveragePower.flatten(),
                   mode="lines",
                   line=dict(color="#000000", width=4),
                   hovertemplate="%{y}<br>Date: %{x|%I:%M %p} <extra></extra>",
                   showlegend=False,
                   legendgroup="averageLFPTrend"
                   )
        )

    fig.update_yaxes(type="linear",
                     title_font_size=15, title_text="Power (a.u.)", tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     zeroline=True, zerolinecolor="#DDDDDD")

    fig.update_xaxes(type="linear", range=(0,24),
                     #tickformat="%H:%M",
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD",
                     zeroline=True, zerolinecolor="#DDDDDD")

    fig.layout["paper_bgcolor"]="#FFFFFF"
    fig.layout["plot_bgcolor"]="#FFFFFF"

    fig.update_layout(title={"text": ConfigurationText,
                            "x": 0.5,
                            "font": dict(size=18)})

    return  po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div')

def getAveragePSDEvents(Data, channelName):
    if "EventPSDGroup" in Data.keys():
        if channelName in Data["EventPSDGroup"].keys():
            fig = go.Figure()
            cmap = plt.get_cmap("Set1", 9)
            Frequency = np.array(range(100))*250/256
            for eventName in Data["EventPSDGroup"][channelName].keys():
                for i in range(len(Data["EventPSDGroup"]["UniqueEventName"])):
                    if Data["EventPSDGroup"]["UniqueEventName"][i] == eventName:
                        eventID = i

                fig = addShadedErrorBar(fig, Frequency,
                                            np.mean(Data["EventPSDGroup"][channelName][eventName],axis=0).flatten(),
                                            SignalProcessingUtility.stderr(Data["EventPSDGroup"][channelName][eventName],axis=0).flatten(),
                                            color=colorTextFromCmap(cmap(eventID)),
                                            alpha=0.3,
                                            legendgroup=eventName + f" (n={Data['EventPSDGroup'][channelName][eventName].shape[0]})")
                fig.add_trace(
                        go.Scatter(x=Frequency,
                                   y=np.mean(Data["EventPSDGroup"][channelName][eventName],axis=0),
                                   mode="lines",
                                   line=dict(color=colorTextFromCmap(cmap(eventID)), width=2),
                                   name=eventName + f" (n={Data['EventPSDGroup'][channelName][eventName].shape[0]})",
                                   legendgroup=eventName + f" (n={Data['EventPSDGroup'][channelName][eventName].shape[0]})",
                                   hovertemplate=eventName + " Power: %{y} <extra></extra>",
                                   showlegend=True,
                                   )
                        )

            defaultYAxis = np.logspace(-2,1,num=4)
            fig.update_yaxes(type="log", range=(-2, 1),
                             title_font_size=15, title_text="Power (a.u.)", tickfont_size=12,
                             tickmode="array", tickvals=defaultYAxis, ticktext=[f"{i}" for i in defaultYAxis],
                             color="#000000", ticks="outside", showline=True, linecolor="#000000",
                             showgrid=True, gridcolor="#DDDDDD",
                             zeroline=True, zerolinecolor="#DDDDDD")

            fig.update_xaxes(type="linear", range=(0, 100),
                             color="#000000", ticks="outside", showline=True, linecolor="#000000",
                             showgrid=True, gridcolor="#DDDDDD",
                             zeroline=True, zerolinecolor="#DDDDDD")

            fig.layout["paper_bgcolor"]="#FFFFFF"
            fig.layout["plot_bgcolor"]="#FFFFFF"
            fig.update_layout(hovermode="x")
            fig.update_layout(legend=dict( yanchor="top", y=0.99, xanchor="left", x=0.7 ),
                                title=dict( text=channelName, x=0.5, font=dict(size=18)))
            return po.plot(fig, auto_open=False, include_plotlyjs=False, output_type='div')

    return ""
