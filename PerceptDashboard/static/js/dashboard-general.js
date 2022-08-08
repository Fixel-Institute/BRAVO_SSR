function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
const csrftoken = getCookie('csrftoken');

function showNotification(type, from, align, message, delay) {
  if (type == "success") icon = "ni ni-like-2"
  else if (type == "danger") icon = "ni ni-bell-55"
  else icon = ""

  $.notify({
    icon: icon ,
    message: message
  }, {
    type: type,
    timer: delay,
    placement: {
      from: from,
      align: align
    }
  });
}

if (!String.prototype.format) {
  String.prototype.format = function() {
    var args = arguments;
    return this.replace(/{(\d+)}/g, function(match, number) {
      return typeof args[number] != 'undefined'
        ? args[number]
        : match
      ;
    });
  };
}

function matchArray(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i])
}

function getTimestring(timestamp)
{
  var d = new Date(timestamp);
  d.setTime( d.getTime() + d.getTimezoneOffset()*60*1000 );
  return d
}

function downSample(array, skips)
{
  newArray = []
  for (i = 0; i < array.length; i += skips) newArray.push(array[i])
  return newArray
}

function formatDateString(dateStruct, formatter)
{
  MonthString = ["January","Feburary","March","April","May","June","July","August","September","October","November","December"]
  MonthString_Short = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]
  if (dateStruct.getHours() > 11) ampm = "PM"
  else ampm = "AM"
  formatter = formatter.replaceAll("{%Y}",dateStruct.getFullYear()).replaceAll("{%B}",MonthString[dateStruct.getMonth()]).replaceAll("{%b}",MonthString_Short[dateStruct.getMonth()]).replaceAll("{%D}",dateStruct.getDate())
  formatter = formatter.replaceAll("{%H}",dateStruct.getHours()).replaceAll("{%M}",String(dateStruct.getMinutes()).padStart(2,'0')).replaceAll("{%S}",String(dateStruct.getSeconds()).padStart(2,'0')).replaceAll("{%P}",ampm)
  return formatter
}

function getColorString(v,vmin,vmax,alpha)
{
   var dv;
   var color = [1,1,1]

   if (v < vmin) v = vmin;
   if (v > vmax) v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      color[0] = 0;
      color[1] = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      color[0] = 0;
      color[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      color[0] = 4 * (v - vmin - 0.5 * dv) / dv;
      color[2] = 0;
   } else {
      color[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      color[2] = 0;
   }

   var rgb = "rgb(" + Math.round(color[0]*255) + "," + Math.round(color[1]*255) + "," + Math.round(color[2]*255) + ")"

   return rgb;
}

function formatSegmentString(channels) {
  var channelName = "";
  for (var i = 0; i < channels.length; i++) {
    switch (channels[i]) {
      case 0:
        channelName += "E0"
        break
      case 1:
        channelName += "E1"
        break
      case 1.1:
        channelName += "E1A"
        break
      case 1.2:
        channelName += "E1B"
        break
      case 1.3:
        channelName += "E1C"
        break
      case 2:
        channelName += "E2"
        break
      case 2.1:
        channelName += "E2A"
        break
      case 2.2:
        channelName += "E2B"
        break
      case 2.3:
        channelName += "E2C"
        break
      case 3:
        channelName += "E3"
        break
    }

    if (i == 0) {
      channelName += " - "
    }
  }
  return channelName;
}
