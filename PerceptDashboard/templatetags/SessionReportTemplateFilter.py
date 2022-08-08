from django import template
import datetime
import numpy as np

register = template.Library()

@register.filter
def updateDateOfBirth(value):
    try:
        dob = datetime.datetime.fromisoformat(value+"+00:00").strftime("%Y/%m/%d")
        return dob
    except:
        return ""

@register.filter
def formatGroupId(groupId):
    if groupId == "GroupIdDef.GROUP_A":
        return "Group A"
    elif groupId == "GroupIdDef.GROUP_B":
        return "Group B"
    elif groupId == "GroupIdDef.GROUP_C":
        return "Group C"
    elif groupId == "GroupIdDef.GROUP_D":
        return "Group D"
    return groupId

@register.filter
def formatBatteryString(batteryLife):
    if type(batteryLife) == int:
        YearLeft = int(np.floor(batteryLife / 12))
        MonthLeft = batteryLife % 12
        if YearLeft > 1:
            batteryString = f"{YearLeft} years "
        else:
            batteryString = f"{YearLeft} year "

        if MonthLeft > 1:
            return batteryString + f" {MonthLeft} months"
        else:
            return batteryString + f" {MonthLeft} month"
    else:
        return ""

@register.filter
def formatLocationString(location):
    return location.replace("InsLocation.","").replace("_"," ").title()

@register.filter
def formatLeadNames(leadConfig):
    if leadConfig == "InsPort.ZERO_THREE":
        return "E00-E03"
    elif leadConfig == "InsPort.ZERO_SEVEN":
        return "E00-E07"
    elif leadConfig == "InsPort.EIGHT_ELEVEN":
        return "E08-E11"
    elif leadConfig == "InsPort.EIGHT_FIFTEEN":
        return "E08-E15"
    elif leadConfig == "LeadModelDef.LEAD_B33015":
        return "SenSight B33015"
    elif leadConfig == "LeadModelDef.LEAD_B33005":
        return "SenSight B33005"
    elif leadConfig == "LeadModelDef.LEAD_3387":
        return "Medtronic 3387"
    elif leadConfig == "LeadModelDef.LEAD_3389":
        return "Medtronic 3389"
    elif leadConfig == "LeadLocationDef.Vim":
        return "VIM"
    elif leadConfig == "LeadLocationDef.Stn":
        return "STN"
    elif leadConfig == "LeadLocationDef.Gpi":
        return "GPi"
    elif leadConfig == "LeadLocationDef.Other":
        return "Other"
    elif leadConfig.startswith("HemisphereLocationDef."):
        return leadConfig.replace("HemisphereLocationDef.","") + " "
    return leadConfig

@register.filter
def formatChannelName(contact):
    try:
        if contact["ElectrodeStateResult"] != "ElectrodeStateDef.None":
            if contact["Electrode"] == "ElectrodeDef.Case":
                electrodeName = "C"
            elif contact["Electrode"].find("ElectrodeDef.FourElectrodes_") >= 0:
                electrodeName = "E" + contact["Electrode"].replace("ElectrodeDef.FourElectrodes_","")
            elif contact["Electrode"].find("ElectrodeDef.Sen") >= 0:
                electrodeName = "E" + contact["Electrode"].upper().replace("ELECTRODEDEF.SENSIGHT_","")

            if "ElectrodeAmplitudeInMilliAmps" in contact.keys():
                fractionAmplitude = f" ({contact['ElectrodeAmplitudeInMilliAmps']:.2f}mA)"
            else:
                fractionAmplitude = ""

            if contact["ElectrodeStateResult"] == "ElectrodeStateDef.Negative":
                return electrodeName + "+" + fractionAmplitude
            else:
                return electrodeName + "-" + fractionAmplitude
        return ""
    except:
        return contact

@register.filter
def precision(value, arg):
    arg = int(arg)
    if arg == 1:
        return f"{np.around(value, arg):.1f}"
    elif arg == 2:
        return f"{np.around(value, arg):.2f}"
    elif arg == 3:
        return f"{np.around(value, arg):.3f}"

    return f"{np.around(value, arg)}"
